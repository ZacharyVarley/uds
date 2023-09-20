import torch
from FastGeodis import generalised_geodesic2d
import torch.nn.functional as nnf
from scipy.spatial import cKDTree


@torch.jit.script
def _local_variance(mask: torch.Tensor,
                    img: torch.Tensor,
                    img2: torch.Tensor,
                    vwr: int,
                    vthresh: float,
                    ) -> torch.Tensor:
    
    # Calculate integral images with padding
    padding = (vwr + 1, vwr, vwr + 1, vwr)
    img_int = nnf.pad(img.double(), padding).cumsum(dim=-2).cumsum(dim=-1)
    img2_int = nnf.pad(img2.double(), padding).cumsum(dim=-2).cumsum(dim=-1)
    mask_int = nnf.pad(mask.double(), padding).cumsum(dim=-2).cumsum(dim=-1)

    H, W = img.shape[-2:]

    # Compute terms for local variance
    local_sum = (
        img_int[..., 2*vwr + 1:H + 2*vwr + 1, 2*vwr + 1:W + 2*vwr + 1]
        - img_int[..., 2*vwr + 1:H + 2*vwr + 1, :W]
        - img_int[..., :H, 2*vwr + 1:W + 2*vwr + 1]
        + img_int[..., :H, :W]
    )

    local_sum_squared = (
        img2_int[..., 2*vwr + 1:H + 2*vwr + 1, 2*vwr + 1:W + 2*vwr + 1]
        - img2_int[..., 2*vwr + 1:H + 2*vwr + 1, :W]
        - img2_int[..., :H, 2*vwr + 1:W + 2*vwr + 1]
        + img2_int[..., :H, :W]
    )

    local_counts = (
        mask_int[..., 2*vwr + 1:H + 2*vwr + 1, 2*vwr + 1:W + 2*vwr + 1]
        - mask_int[..., 2*vwr + 1:H + 2*vwr + 1, :W]
        - mask_int[..., :H, 2*vwr + 1:W + 2*vwr + 1]
        + mask_int[..., :H, :W]
    )

    # Calculate local variance
    local_variance = torch.ones_like(img)

    valid = local_counts > 2
    valid_counts = local_counts[valid]

    local_mean = local_sum[valid] / valid_counts
    local_mean_squared = local_sum_squared[valid] / valid_counts
    local_variance[valid] = (local_mean_squared - local_mean ** 2).float()

    # variance less than factor squared is zero
    local_variance[local_variance < vthresh] = 0.0

    return local_variance


@torch.jit.script
def _argrelmax(scores: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Find the relative maxima in a 2D tensor.

    Parameters:
        a (torch.Tensor): The input 2D tensor.
        radius (int): The radius of the max pooling window.

    Returns:
        torch.Tensor: A tensor containing the (row, col) coordinates of the relative maxima.
    """

    # Calculate full window size based on radius
    width = 2 * radius + 1

    # Perform max pooling and also get the indices of max elements
    window_maxima, indices = torch.nn.functional.max_pool2d_with_indices(
        scores, width, stride=1, padding=radius
    )

    # Remove singleton dimensions
    window_maxima.squeeze_()
    indices.squeeze_()

    # Finding the unique indices of maxima
    candidates = torch.unique(indices)

    # Finding "nice peaks" or good local maxima
    nice_peaks = candidates[
        (indices.view(-1)[candidates] == candidates).nonzero()
    ].squeeze()

    # Convert these flat indices back to 2D indices
    _, cols = scores.shape[-2:]
    nice_peaks_2d = torch.stack((nice_peaks // cols, nice_peaks % cols), dim=1)

    return nice_peaks_2d


@torch.jit.script
def _local_maxima(scores: torch.Tensor, radius: int):
    """
    Perform non-maximal suppression on a 2D tensor. Return the coordinates of the
    local maxima.
    """
    # Calculate full window size based on radius
    width = 2 * radius + 1

    H, W = scores.shape[-2:]

    # use max pooling
    window_maxima, indices = torch.nn.functional.max_pool2d_with_indices(
        scores, width, stride=1, padding=radius
    )

    # Remove singleton dimensions
    window_maxima.squeeze_()
    indices.squeeze_()

    # Finding the unique indices of maxima (these are flattened indices)
    candidates, candidate_indices = torch.unique(indices, return_inverse=True)

    # return the 2D coordiantes of the maxima, and the maxima values
    coords_2d = torch.stack((candidates // W, candidates % W), dim=1)

    return coords_2d #window_maxima[coords_2d[:, 0], coords_2d[:, 1]]


class FastDynamicSampling:
    """
    This is a dynamic sampling method that uses the local neighborhood of the
    image to determine the best pixel locations to sample. Scores are calculated
    for each pixel location as the product of the Euclidean distance and the
    local variance within a set window size. Each iteration a set fraction of
    local maxima are sampled.

    This module uses an ask and tell interface. The ask method returns the
    coordinates of the pixels to sample. The tell method returns the pixel
    values at the coordinates sampled. The ask and tell methods are called
    iteratively until the desired fraction of the image is sampled. So the
    image is not ever given to the module.

    Parameters
    ----------
    device : torch.device
        The device to use.
    region_shape : tuple
        The shape of the region to sample.
    sample_fraction : float
        The fraction of the image to sample.
    var_window_radius : int
        The radius of the window to calculate the local variance.
    nms_window_radius : int
        The radius of the window to perform non-maximum suppression.
    variance_threshold : float
        The threshold for the local variance. If the local variance is less than
        this value, the pixel is considered zero.

    """

    def __init__(
        self,
        device,
        region_shape,
        sample_fraction,
        var_window_radius=10,
        nms_window_radius=2,
        variance_threshold=0.0005,
    ):
        self.region_shape = region_shape
        self.var_window_radius = var_window_radius
        self.nms_window_radius = nms_window_radius
        self.sample_fraction = sample_fraction
        self.variance_threshold = variance_threshold

        # mask to store the pixels sampled
        self.mask = torch.zeros(self.region_shape, dtype=torch.bool, device=device)[None, None]

        # image to store the pixel values
        self.image = torch.zeros(self.region_shape, dtype=torch.float32, device=device)[None, None]

        # x^2 image for calculating the local variance
        self.image_squared = torch.zeros_like(self.image)

        # variable for number of pixels sampled
        self.num_sampled = 0

        # number of batches
        self.num_batches = 0

        # make indicator tensor fed to geodesic distance transform
        self.indicator = torch.ones_like(self.image)

    def ask(self):
        """
        Returns the coordinates of the pixels to sample.

        Returns
        -------
        coords : np.ndarray
            The coordinates of the pixels to sample.

        """
        # Calculate the local variance
        local_var = _local_variance(
            self.mask, self.image, self.image_squared, self.var_window_radius, self.variance_threshold
        )

        # the installation script for FastGeodis is broken for GPU so just cast to CPU
        # this is not a bottleneck anyway
        distance_map = generalised_geodesic2d(
            self.indicator.cpu(), ~self.mask.cpu(), 1e10, 0.0, 1
        ).to(self.image.device)

        # distance_map = generalised_geodesic2d(
        #     self.indicator, ~self.mask, 1e10, 0.0, 1
        # )

        # import matplotlib.pyplot as plt

        # # plot distance map
        # plt.figure(figsize=(10, 10))
        # plt.imshow(distance_map.squeeze().cpu().numpy(), cmap="gray")
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(f"test_img_dist_{self.num_batches}.png")

        # # plot variance map
        # plt.figure(figsize=(10, 10))
        # plt.imshow(local_var.squeeze().cpu().numpy(), cmap="gray")
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(f"test_img_var_{self.num_batches}.png")
        
        # Calculate the scores
        scores = distance_map * local_var

        # set the scores of the pixels already sampled to negative infinity
        scores[self.mask] = -torch.inf

        # Perform non-maximum suppression
        # coords = _local_maxima(scores, self.nms_window_radius)
        coords = _argrelmax(scores, self.nms_window_radius)
        # coords = _find_peaks(scores, self.nms_window_radius, self.region_shape)

        # correctly determine the number of pixels to sample
        num_to_sample = min(int(self.sample_fraction * self.region_shape[0] * self.region_shape[1] - self.num_sampled), len(coords))

        # sort the scores and return the top fraction
        best_scores = scores[0, 0, coords[:, 0], coords[:, 1]]
        sorted_indices = torch.argsort(best_scores, descending=True)
        coords = coords[sorted_indices[:num_to_sample]]

        # Return the coordinates
        return coords

    def tell(self, coords, values):
        """
        Returns the pixel values at the coordinates sampled.

        Parameters
        ----------
        coords : np.ndarray
            The coordinates of the pixels sampled.
        values : np.ndarray
            The pixel values at the coordinates sampled.

        """
        # Update the image
        self.image[0, 0, coords[:, 0], coords[:, 1]] = values

        # Update the image squared
        self.image_squared[0, 0, coords[:, 0], coords[:, 1]] = (values)**2

        # Update the mask
        self.mask[0, 0, coords[:, 0], coords[:, 1]] = True

        # Update the number of pixels sampled
        self.num_sampled += len(coords)

        # Update the number of batches
        self.num_batches += 1

    def reset(self):
        """
        Reset the sampler.

        """
        self.mask = torch.zeros(self.region_shape, dtype=torch.bool, device=self.image.device)[None, None]
        self.image = torch.zeros(self.region_shape, dtype=torch.float32, device=self.image.device)[None, None]
        self.image_squared = torch.zeros_like(self.image)
        self.num_sampled = 0
        self.num_batches = 0


    def impute_kdtree(
            self, 
            k: int = 4,
    ):
        # make a KDTree of the sampled pixels
        coords = torch.stack(torch.where(self.mask.squeeze())).T.cpu().numpy()
        values = self.image.squeeze()[self.mask.squeeze()].cpu().numpy()
        kdtree = cKDTree(coords)

        # find the pixels to impute
        impute_coords = torch.stack(torch.where(~self.mask.squeeze())).T.cpu().numpy()
        distances, indices = kdtree.query(impute_coords, k=k)

        # impute the pixels with the inverse distance weighted mean
        impute_values = (values[indices] / distances).sum(axis=1) / (1 / distances).sum(axis=1)
        
        output_image = self.image.clone()
        output_image[0, 0, impute_coords[:, 0], impute_coords[:, 1]] = torch.from_numpy(impute_values).to(self.image.device).to(self.image.dtype)

        return output_image
