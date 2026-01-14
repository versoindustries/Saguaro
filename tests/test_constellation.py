import unittest
import shutil
import tempfile
import os
from saguaro.constellation.manager import ConstellationManager


class TestConstellationManager(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the constellation
        self.test_dir = tempfile.mkdtemp()
        self.constellation_path = os.path.join(
            self.test_dir, ".saguaro", "constellation"
        )
        self.manager = ConstellationManager(base_path=self.constellation_path)

        # Create a dummy library to index
        self.dummy_lib_path = os.path.join(self.test_dir, "my_lib")
        os.makedirs(self.dummy_lib_path)
        with open(os.path.join(self.dummy_lib_path, "lib.py"), "w") as f:
            f.write("print('hello world')")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        self.assertTrue(os.path.exists(self.constellation_path))

    def test_index_library(self):
        success = self.manager.index_library("my_lib_v1", self.dummy_lib_path)
        self.assertTrue(success)

        libs = self.manager.list_libraries()
        self.assertIn("my_lib_v1", libs)

        # Verify metadata
        meta_path = os.path.join(self.constellation_path, "my_lib_v1", "meta.yaml")
        self.assertTrue(os.path.exists(meta_path))

    def test_link_to_project(self):
        # First index a lib
        self.manager.index_library("shared_lib", self.dummy_lib_path)

        # Create a dummy project saguaro dir
        project_saguaro = os.path.join(self.test_dir, "project", ".saguaro")
        os.makedirs(project_saguaro)

        # Link
        success = self.manager.link_to_project("shared_lib", project_saguaro)
        self.assertTrue(success)

        # Verify link
        link_path = os.path.join(project_saguaro, "links", "shared_lib")
        self.assertTrue(os.path.exists(link_path))
        with open(link_path, "r") as f:
            content = f.read()
        self.assertIn("pointer:", content)


if __name__ == "__main__":
    unittest.main()
