import matplotlib.pyplot as plt
import numpy as np

from diffusion_nav.collision import Rectangle
from diffusion_nav.env import NavigationEnv
from diffusion_nav.kinematics import Pose2D


def main() -> None:
    """Run a short deterministic rollout and render the resulting environment state."""
    environment = NavigationEnv(
        width=10.0,
        height=8.0,
        robot_radius=0.25,
        start_pose=Pose2D(x=1.0, y=1.0, theta=0.0),
        goal=(9.0, 7.0),
        obstacles=[
            Rectangle(xmin=3.0, ymin=2.0, xmax=4.0, ymax=5.0),
            Rectangle(xmin=6.0, ymin=4.0, xmax=7.0, ymax=6.0),
        ],
    )
    environment.reset()

    test_action = np.array([0.5, 0.1], dtype=np.float32)
    for _ in range(10):
        environment.step(test_action)

    figure, _ = environment.render("artifacts/gate3_map.png")
    plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
