import torch

from uds.orientations import quaternion_multiply, quaternion_real_of_prod, quaternion_apply

# sqrt(2) / 2 and sqrt(3) / 2
R2 = 0.7071067811865475244008443621048490392848359376884740365883398689
R3 = 0.8660254037844386467637231707529361834714026269051903140279034897

# 7 subsets of O Laue groups (O, T, D4, D2, C4, C2, C1)
LAUE_O = torch.tensor(
    [
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [R2, 0, 0, R2],
        [R2, 0, 0, -R2],
        [0, R2, R2, 0],
        [0, -R2, R2, 0],
        [0.5, 0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5, -0.5],
        [0.5, -0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
        [R2, R2, 0, 0],
        [R2, -R2, 0, 0],
        [R2, 0, R2, 0],
        [R2, 0, -R2, 0],
        [0, R2, 0, R2],
        [0, -R2, 0, R2],
        [0, 0, R2, R2],
        [0, 0, -R2, R2],
    ],
    dtype=torch.float64,
)
LAUE_T = torch.tensor(
    [
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0.5, 0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5, -0.5],
        [0.5, -0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
    ],
    dtype=torch.float64,
)
LAUE_D4 = torch.tensor(
    [
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [R2, 0, 0, R2],
        [R2, 0, 0, -R2],
        [0, R2, R2, 0],
        [0, -R2, R2, 0],
    ],
    dtype=torch.float64,
)
LAUE_D2 = torch.tensor(
    [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.float64
)
LAUE_C4 = torch.tensor(
    [[1, 0, 0, 0], [0, 0, 0, 1], [R2, 0, 0, R2], [R2, 0, 0, -R2]], dtype=torch.float64
)
LAUE_C2 = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.float64)
LAUE_C1 = torch.tensor([[1, 0, 0, 0]], dtype=torch.float64)


# subsets of D6 Laue groups (D6, D3, C6, C3) - C1 was already defined above
LAUE_D6 = torch.tensor(
    [
        [1, 0, 0, 0],
        [0.5, 0, 0, R3],
        [0.5, 0, 0, -R3],
        [0, 0, 0, 1],
        [R3, 0, 0, 0.5],
        [R3, 0, 0, -0.5],
        [0, 1, 0, 0],
        [0, -0.5, R3, 0],
        [0, 0.5, R3, 0],
        [0, R3, 0.5, 0],
        [0, -R3, 0.5, 0],
        [0, 0, 1, 0],
    ],
    dtype=torch.float64,
)
LAUE_D3 = torch.tensor(
    [
        [1, 0, 0, 0],
        [0.5, 0, 0, R3],
        [0.5, 0, 0, -R3],
        [0, 1, 0, 0],
        [0, -0.5, R3, 0],
        [0, 0.5, R3, 0],
    ],
    dtype=torch.float64,
)
LAUE_C6 = torch.tensor(
    [
        [1, 0, 0, 0],
        [0.5, 0, 0, R3],
        [0.5, 0, 0, -R3],
        [0, 0, 0, 1],
        [R3, 0, 0, 0.5],
        [R3, 0, 0, -0.5],
    ],
    dtype=torch.float64,
)
LAUE_C3 = torch.tensor(
    [[1, 0, 0, 0], [0.5, 0, 0, R3], [0.5, 0, 0, -R3]], dtype=torch.float64
)

LAUE_GROUPS = [
    LAUE_C1,
    LAUE_C2,
    LAUE_C3,
    LAUE_C4,
    LAUE_C6,
    LAUE_D2,
    LAUE_D3,
    LAUE_D4,
    LAUE_D6,
    LAUE_T,
    LAUE_O,
]

LAUE_MULTS = [
    2,
    4,
    6,
    8,
    12,
    8,
    12,
    16,
    24,
    24,
    48,
]

@torch.jit.script
def _ori_to_so3_fz(
    orientations: torch.Tensor, laue_group: torch.Tensor
) -> torch.Tensor:
    """
    :param misorientations: quaternions to move to fundamental zone of shape (..., 4)
    :param laue_group: laue group of quaternions to move to fundamental zone
    :return: orientations in fundamental zone of shape (..., 4)
    """
    # get the important shapes
    data_shape = orientations.shape
    N = torch.prod(torch.tensor(data_shape[:-1]))
    card = laue_group.shape[0]

    # reshape so that quaternions is (N, 1, 4) and laue_group is (1, card, 4) then use broadcasting
    equivalent_quaternions_real = quaternion_real_of_prod(
        orientations.reshape(N, 1, 4), laue_group.reshape(card, 4)
    ).abs()

    # find the quaternion with the largest w value
    row_maximum_indices = torch.argmax(equivalent_quaternions_real, dim=-1)

    # gather the equivalent quaternions with the largest w value for each equivalent quaternion set
    output = quaternion_multiply(
        orientations.reshape(N, 4), laue_group[row_maximum_indices]
    )

    return output.reshape(data_shape)


@torch.jit.script
def _oris_are_in_so3_fz(orientations: torch.Tensor, laue_group: torch.Tensor) -> torch.Tensor:
    """
    :param misorientations: quaternions to move to fundamental zone of shape (..., 4)
    :param laue_group: laue group of quaternions to move to fundamental zone
    :return: orientations in fundamental zone of shape (..., 4)
    """
    # get the important shapes
    data_shape = orientations.shape
    N = torch.prod(torch.tensor(data_shape[:-1]))
    card = laue_group.shape[0]

    # reshape so that quaternions is (N, 1, 4) and laue_group is (1, card, 4) then use broadcasting
    equivalent_quaternions_real = quaternion_real_of_prod(
        orientations.reshape(N, 1, 4), laue_group.reshape(card, 4)
    ).abs()

    # find the quaternion with the largest w value
    row_maximum_indices = torch.argmax(equivalent_quaternions_real, dim=-1)

    # first element is always the identity so if the index is 0, then it is in the fundamental zone
    return (row_maximum_indices == 0).reshape(data_shape[:-1])


def oris_are_in_so3_fz(orientations: torch.Tensor, laue_group: int) -> torch.Tensor:
    """
    :param orientations: quaternions to move to fundamental zone of shape (..., 4)
    :param laue_group: laue group of quaternions to move to fundamental zone
    :return: orientations in fundamental zone of shape (..., 4)
    """
    # assert LAUE_GROUP is an int between 1 and 11 inclusive
    if not isinstance(laue_group, int) or laue_group < 1 or laue_group > 11:
        raise ValueError(f"Laue group {laue_group} not laue number in [1, 11]")
    # find all equivalent quaternions
    return _oris_are_in_so3_fz(
        orientations,
        LAUE_GROUPS[laue_group - 1].to(orientations.dtype).to(orientations.device),
    )


def oris_to_so3_fz(quaternions: torch.Tensor, laue_group: int) -> torch.Tensor:
    """
    :param quaternions: quaternions to move to fundamental zone of shape (..., 4)
    :param laue_group: laue group of quaternions to move to fundamental zone
    :return: orientations in fundamental zone of shape (..., 4)
    """
    # assert LAUE_GROUP is an int between 1 and 11 inclusive
    if not isinstance(laue_group, int) or laue_group < 1 or laue_group > 11:
        raise ValueError(f"Laue group {laue_group} not laue number in [1, 11]")
    # find all equivalent quaternions
    return _ori_to_so3_fz(
        quaternions,
        LAUE_GROUPS[laue_group - 1].to(quaternions.dtype).to(quaternions.device),
    )

