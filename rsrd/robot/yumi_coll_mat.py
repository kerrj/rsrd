yumi_links = [
    "world",
    "base_link",
    "yumi_link_1_r",
    "yumi_link_2_r",
    "yumi_link_3_r",
    "yumi_link_4_r",
    "yumi_link_5_r",
    "yumi_link_6_r",
    "yumi_link_7_r",
    "yumi_link_1_l",
    "yumi_link_2_l",
    "yumi_link_3_l",
    "yumi_link_4_l",
    "yumi_link_5_l",
    "yumi_link_6_l",
    "yumi_link_7_l",
    "right_dummy_point",
    "right_suction_point",
    "gripper_r_base",
    "gripper_r_finger_r",
    "gripper_r_finger_l",
    "left_dummy_point",
    "left_gripper_point",
    "gripper_l_base",
    "gripper_l_finger_r",
    "gripper_l_finger_l",
]

yumi_arm_r = [
    "yumi_link_1_r",
    "yumi_link_2_r",
    "yumi_link_3_r",
    "yumi_link_4_r",
    "yumi_link_5_r",
    "yumi_link_6_r",
    "yumi_link_7_r",
    "gripper_r_base",
    "gripper_r_finger_r",
    "gripper_r_finger_l",
]

yumi_arm_l = [
    "yumi_link_1_l",
    "yumi_link_2_l",
    "yumi_link_3_l",
    "yumi_link_4_l",
    "yumi_link_5_l",
    "yumi_link_6_l",
    "yumi_link_7_l",
    "gripper_l_base",
    "gripper_l_finger_r",
    "gripper_l_finger_l",
]

world = [
    "world",
    "base_link",
    "right_dummy_point",
    "left_dummy_point",
    "right_suction_point",
    "right_gripper_point",
    "left_suction_point",
    "left_gripper_point",
]

self_coll_ignore_pairs = [
    *[
        (link1, link2)
        for link1 in yumi_arm_r
        for link2 in yumi_arm_r
    ],
    *[
        (link1, link2)
        for link1 in yumi_arm_l
        for link2 in yumi_arm_l
    ],
    *[
        (link1, link2)
        for link1 in world
        for link2 in world
    ],
    *[
        (link1, link2)
        for link1 in yumi_arm_r
        for link2 in world
    ],
    *[
        (link1, link2)
        for link1 in yumi_arm_l
        for link2 in world
    ],
    *[
        (link1, link2)
        for link1 in yumi_arm_r[:4]
        for link2 in yumi_arm_l[:4]
    ]
]