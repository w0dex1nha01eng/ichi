import matplotlib.pyplot as plt

from diffusion_nav.collision import Rectangle
from diffusion_nav.env import NavigationEnv
from diffusion_nav.kinematics import Pose2D


env = NavigationEnv(10.0,8.0,0.25,Pose2D(1.0, 1.0, 0.0),(9.0, 7.0),
                    [Rectangle(3.0, 2.0, 4.0, 5.0),Rectangle(6.0, 4.0, 7.0, 6.0)])

env.reset()

for _ in range(10):
    env.step([0.5, 0.1])

fig, ax = env.render("artifacts/gate3_map.png")
plt.show()
plt.close(fig)