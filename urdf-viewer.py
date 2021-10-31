# Copyright 2021 ikeyasu.com
# Parts of this code are inherited from
# https://github.com/google/brax/blob/b99b11109079ff1916673d821ae19baa3132553a/brax/tools/urdf_converter.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# noinspection PyUnresolvedReferences
import gym
# noinspection PyUnresolvedReferences
import brax
from brax import envs
from brax import jumpy as jp
from brax.envs import State, env
from brax.io import html, file
from brax.tools import urdf
from brax.physics import config_pb2
import http.server

from google.protobuf import text_format
from absl import app
from absl import flags
from absl import logging

PORT = 8000
FLAGS = flags.FLAGS

flags.DEFINE_string('xml_model_path', None,
                    'Path of the URDF model to import.')
flags.DEFINE_string('config_path', None, 'Path of the output config.')
flags.DEFINE_bool('add_collision_pairs', False,
                  'Adds the collision pairs to the config.')
# System parameters. See brax/physics/config.proto for more information.
flags.DEFINE_float('angular_damping', -0.05,
                   'Angular velocity damping applied to each body.')
flags.DEFINE_float(
    'baumgarte_erp', 0.1,
    'How aggressively interpenetrating bodies should push away each another.')
flags.DEFINE_float('dt', 0.02, 'Time to simulate each step, in seconds.')
flags.DEFINE_float('friction', 0.6,
                   'How much surfaces in contact resist translation.')
flags.DEFINE_integer('substeps', 4,
                     'Substeps to perform to maintain numerical stability.')

flags.DEFINE_bool('add_floor', True,
                  'Whether or not to add a floor to the scene.')


def convert() -> str:
    filename = FLAGS.xml_model_path
    with file.File(filename) as f:
        logging.info('Loading urdf model from %s', filename)
        xml_string = f.read()

    # Convert the model.
    m = urdf.UrdfConverter(
        xml_string, add_collision_pairs=FLAGS.add_collision_pairs)
    config = m.config

    # Add the default options.
    config.angular_damping = FLAGS.angular_damping
    config.baumgarte_erp = FLAGS.baumgarte_erp
    config.dt = FLAGS.dt
    config.friction = FLAGS.friction
    config.substeps = FLAGS.substeps

    if FLAGS.add_floor:
        floor = config.bodies.add()
        floor.name = 'floor'
        floor.frozen.all = True
        floor.colliders.add(plane=config_pb2.Collider.Plane())
        floor.mass = 1
        floor.inertia.MergeFrom(config_pb2.Vector3(x=1, y=1, z=1))

    return text_format.MessageToString(config)


class MyModel(envs.env.Env):
    def __init__(self, config):
        self.rng = jp.random_prngkey(seed=0)
        super().__init__(config)

    def reset(self, rng: jp.ndarray) -> State:
        self.rng = rng
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jp.random_split(rng, 3)
        qpos = self.sys.default_angle() + jp.random_uniform(
            rng1, (self.sys.num_joint_dof,), -.1, .1)
        qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -.1, .1)
        qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
        obs = jp.zeros(3)
        reward, done, zero = jp.zeros(3)
        metrics = {
            'reward_ctrl_cost': zero,
            'reward_contact_cost': zero,
            'reward_forward': zero,
            'reward_survive': zero,
        }
        # noinspection PyArgumentList
        return env.State(qp, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        return self.reset(self.rng)


# noinspection PyUnusedLocal
def main(argv):
    e = MyModel(convert())
    state = e.reset(rng=jp.random_prngkey(seed=0))

    class MyHTTPHandler(http.server.BaseHTTPRequestHandler):
        # noinspection PyPep8Naming
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            h = html.render(e.sys, [state.qp])
            self.wfile.write(h.encode())

    server_address = ('', PORT)
    httpd = http.server.HTTPServer(server_address, MyHTTPHandler)
    print("Please open http://127.0.0.1:" + str(PORT) + "/ in your browser")
    httpd.serve_forever()


if __name__ == '__main__':
    app.run(main)
