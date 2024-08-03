import os
from conans import ConanFile, tools


class FizzBuzzDeviceTestConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    # VirtualRunEnv can be avoided if "tools.env.virtualenv:auto_use" is defined
    # (it will be defined in Conan 2.0)
    generators = "VirtualRunEnv"
    apply_env = False

    def test(self):
        if not tools.cross_building(self):
            self.run("fizz_buzz_device", env="conanrun")
