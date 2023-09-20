import torch
from scipy.spatial import cKDTree
import time

from uds.sampler import FastDynamicSampling


def simulated_sample_ebsd_volume(
    quats: torch.Tensor,
    fraction_starting: float,
    fraction_sampled: float,
    win_max_radius: int,
    win_var_radius: int,
    vthresh: float,
    device: torch.device,
    sobol_seed: bool = True,
    scales: tuple[float, float, float] = (1.0, 1.0, 1.0),
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
        image = image[..., 0]
        image = (image - image.min()) / (image.max() - image.min())

        if sobol_seed:
            # generate Sobol sequence sampling of 5% of the image
            sobol = torch.quasirandom.SobolEngine(2, scramble=True)
            rand_coords = sobol.draw(n_starting).to(device)
            rand_coords[:, 0] *= (H - 1)
            rand_coords[:, 1] *= (W - 1)
            rand_coords = rand_coords.long()
        else:
            # Sample 1% of the image with random initialisation
            rand_flat_inds = torch.randperm(H * W).to(device)[:n_starting]
            rand_coords = torch.stack((rand_flat_inds // W, rand_flat_inds % W), dim=1)

        # Tell the sampler the initialisation
        sampler.tell(rand_coords, image[rand_coords[:, 0], rand_coords[:, 1]])

        # Sample 10% of the image
        while sampler.num_sampled < int(fraction_sampled * H * W):
            coords = sampler.ask()
            sampler.tell(coords, image[coords[:, 0], coords[:, 1]])
            n_samples = sampler.num_sampled
            print(
                f"Layer {s}, N and frac sampled: {n_samples}, {n_samples / (H * W) * 100:.4f}%"
            )

        # enter the sampled pixels into the volume
        sampled_mask[s, :, :] = sampler.mask.squeeze()

        # update the number of dynamic samples
        number_dynamic_samples += sampler.num_sampled - n_starting

    end = time.time()

    us_per_pixel = (end - start) / number_dynamic_samples * 1e6

    # use nearest neighbors to fill the unsampled pixels in the output volume
    output_volume = torch.empty_like(quats)

    # get the indices of the unsampled pixels
    us_inds = torch.where(sampled_mask == False)

    # get the indices of the sampled pixels
    s_inds = torch.where(sampled_mask == True)

    # fill in the sampled pixels
    output_volume[s_inds[0], s_inds[1], s_inds[2], :] = quats[
        s_inds[0], s_inds[1], s_inds[2], :
    ]

    # find the nearest neighbors of the unsampled pixels
    unsampled_coords = torch.stack((us_inds[0] * scales[0], us_inds[1] * scales[1], us_inds[2] * scales[2]), dim=1)
    sampled_coords = torch.stack((s_inds[0] * scales[0], s_inds[1] * scales[1], s_inds[2] * scales[2]), dim=1)

    # create a KDTree
    tree = cKDTree(sampled_coords.cpu().numpy())

    # query the tree for nearest neighbor
    _, nn_inds = tree.query(unsampled_coords.cpu().numpy(), k=1)

    # fill in the unsampled pixels
    output_volume[us_inds[0], us_inds[1], us_inds[2], :] = quats[
        s_inds[0][nn_inds], s_inds[1][nn_inds], s_inds[2][nn_inds], :
    ]

    return output_volume, sampled_mask, us_per_pixel


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
            # generate Sobol sequence sampling of 5% of the image
            sobol = torch.quasirandom.SobolEngine(2, scramble=True)
            rand_coords = sobol.draw(n_starting).to(device)
            rand_coords[:, 0] *= (H - 1)
            rand_coords[:, 1] *= (W - 1)
            rand_coords = rand_coords.long()
        else:
            # Sample 1% of the image with random initialisation
            rand_flat_inds = torch.randperm(H * W).to(device)[:n_starting]
            rand_coords = torch.stack((rand_flat_inds // W, rand_flat_inds % W), dim=1)

        # Tell the sampler the initialisation
        sampler.tell(rand_coords, image[rand_coords[:, 0], rand_coords[:, 1]])

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