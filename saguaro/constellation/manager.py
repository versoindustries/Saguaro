import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class ConstellationManager:
    """
    Manages the Global Constellation (Shared Knowledge Graph).

    The Constellation is a centralized storage for indexed libraries (e.g., standard library,
    popular packages) that can be shared across multiple local projects to avoid redundant indexing.
    """

    def __init__(self, base_path: Optional[str] = None):
        if base_path:
            self.base_path = Path(base_path)
        else:
            self.base_path = Path.home() / ".saguaro" / "constellation"

        self.ensure_constellation_exists()

    def ensure_constellation_exists(self):
        """Ensure the global constellation directory exists."""
        if not self.base_path.exists():
            self.base_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized Global Constellation at {self.base_path}")

    def index_library(self, lib_name: str, lib_path: str) -> bool:
        """
        Index a library into the global constellation.

        Args:
            lib_name: Name of the library (e.g., 'requests-v2.31').
            lib_path: Path to the library source code.

        Returns:
            True if successful, False otherwise.
        """
        target_dir = self.base_path / lib_name

        if target_dir.exists():
            logger.info(f"Library {lib_name} already exists in Constellation.")
            return True

        logger.info(
            f"Indexing library {lib_name} from {lib_path} into Constellation..."
        )

        # In a real implementation, this would trigger the IndexEngine.
        # For now, we simulate indexing by copying/marking the entry.
        try:
            target_dir.mkdir(parents=True, exist_ok=True)

            # Store metadata
            with open(target_dir / "meta.yaml", "w") as f:
                f.write(f"name: {lib_name}\n")
                f.write(f"source: {lib_path}\n")
                f.write("type: library\n")

            # Simulate index files
            with open(target_dir / "index.bin", "wb") as f:
                f.write(b"DUMMY_INDEX_DATA")

            logger.info(f"Successfully indexed {lib_name} to {target_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to index library {lib_name}: {e}")
            return False

    def list_libraries(self) -> List[str]:
        """List all libraries currently in the Constellation."""
        if not self.base_path.exists():
            return []

        libs = []
        for item in self.base_path.iterdir():
            if item.is_dir() and (item / "meta.yaml").exists():
                libs.append(item.name)
        return sorted(libs)

    def link_to_project(self, lib_name: str, project_saguaro_dir: str) -> bool:
        """
        Create a 'Wormhole Pointer' (symlink or reference) from a project to a Constellation library.

        Args:
            lib_name: Name of the library in Constellation.
            project_saguaro_dir: Path to the project's .saguaro directory.
        """
        lib_path = self.base_path / lib_name
        if not lib_path.exists():
            logger.error(f"Library {lib_name} not found in Constellation.")
            return False

        link_dir = Path(project_saguaro_dir) / "links"
        link_dir.mkdir(parents=True, exist_ok=True)

        target_link = link_dir / lib_name

        try:
            if target_link.exists():
                if target_link.is_symlink() or target_link.is_file():
                    target_link.unlink()
                else:
                    logger.warning(
                        f"Path {target_link} exists and is not a symlink/file. Skipping."
                    )
                    return False

            # Create a pointer file instead of symlink for better cross-platform support in this prototype
            # logic, or use symlink if OS supports it. Let's use a meta pointer file.
            with open(target_link, "w") as f:
                f.write(f"pointer: {lib_path.absolute()}")

            logger.info(f"Linked {lib_name} to {project_saguaro_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to link library: {e}")
            return False
