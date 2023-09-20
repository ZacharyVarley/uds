import torch
from scipy.spatial import cKDTree
import time

from uds.sampler import FastDynamicSampling


def simulated_sample_ebsd_slices(
    quats: torch.Tensor,
    fraction_starting: float,
    fraction_sampled: float,
    win_max_radius: int,
    win_var_radius: int,
    vthresh: float,
    device: torch.device,
    sobol_seed: bool = True,
    infill: bool = False,
):
    # get the shape
    S, H, W, _ = quats.shape

    # calculate the number of voxels to sobol / random sample
    n_starting = int(fraction_starting * H * W)

    # make a sampled volume mask (boolean)
    sampled_mask = torch.zeros((S, H, W), dtype=torch.bool, device=device)

    # Create the sampler
    sampler = FastDynamicSampling(
        device, (H, W), fraction_sampled, win_var_radius, win_max_radius, vthresh
    )

    number_dynamic_samples = 0
    start = time.time()

    for s in range(S):
        # reset the sampler
        sampler.reset()

        # select the image
        image = quats[s, :, :, :]
        # has the quaternions to be the sum
        image = torch.sum(image, dim=-1)
        image = (image - image.min()) / (image.max() - image.min())

        if sobol_seed:
            # Sample with Sobol sequence
            coords_start = sampler.ask_sobol(n_starting)
        else:
            # sample with random initialisation
            coords_start = sampler.ask_random(n_starting)

        # Tell the sampler the initialisation
        sampler.tell(coords_start, image[coords_start[:, 0], coords_start[:, 1]])

        # Sample
        while sampler.num_sampled < int(fraction_sampled * H * W):
            coords = sampler.ask()
            sampler.tell(coords, image[coords[:, 0], coords[:, 1]])

        # enter the sampled pixels into the volume
        sampled_mask[s, :, :] = sampler.mask.squeeze()

        # update the number of dynamic samples
        number_dynamic_samples += sampler.num_sampled - n_starting

    end = time.time()

    us_per_pixel = (end - start) / number_dynamic_samples * 1e6

    if infill:
        # use nearest neighbors to fill the unsampled pixels in the output volume
        output_volume = torch.empty_like(quats)

        print("Starting imputation of unsampled voxels...")
        
        # for each slice individually use the nearest neighbors to fill the unsampled pixels in the output volume
        for s in range(S):
            # get the indices of the unsampled pixels
            us_inds = torch.where(sampled_mask[s, :, :] == False)

            # get the indices of the sampled pixels
            s_inds = torch.where(sampled_mask[s, :, :] == True)

            # fill in the sampled pixels
            output_volume[s, s_inds[0], s_inds[1], :] = quats[
                s, s_inds[0], s_inds[1], :
            ]

            # find the nearest neighbors of the unsampled pixels
            unsampled_coords = torch.stack((us_inds[0], us_inds[1]), dim=1)
            sampled_coords = torch.stack((s_inds[0], s_inds[1]), dim=1)

            # create a KDTree
            tree = cKDTree(sampled_coords.cpu().numpy())

            # query the tree for nearest neighbor
            _, nn_inds = tree.query(unsampled_coords.cpu().numpy(), k=1)

            # fill in the unsampled pixels
            output_volume[s, us_inds[0], us_inds[1], :] = quats[
                s, s_inds[0][nn_inds], s_inds[1][nn_inds], :
            ]
        return output_volume, sampled_mask, us_per_pixel
    
    else:
        return sampled_mask, us_per_pixel