from md2post import *

class ObsidianPage(object):
    """
    Represents a single page (markdown file) in an Obsidian Vault.

    This class is designed to handle the parsing, management, and export of markdown
    files that conform to a specific structure in an Obsidian Vault. It interacts with
    properties and contents of a page, allowing filtering, modification, and exporting
    to other locations.

    Key Components:
    - `Property`: Handles YAML frontmatter (e.g., `blog-post`, `blog-directory`) of the page.
    - `Content`: Manages the content lines of the markdown file (excluding frontmatter).

    Key Features:
    - Reads and parses the markdown file at initialization.
    - Determines if a page is publishable based on the `blog-post` property.
    - Supports exporting the page with modifications to its properties or contents.

    Use Case:
    This class is particularly useful for applications that process Obsidian Vaults to
    generate blog posts, migrate content, or manage markdown files programmatically.
    """
    def __init__(self, page_path: Path):
        """
        Initialize an ObsidianPage instance.

        Arguments:
        - page_path (Path): The path to the markdown file representing the page.

        This method reads the markdown file and initializes its properties (`prpts`) and contents (`contents`).
        """
        self.page_path = page_path.resolve()
        with open(self.page_path, "r") as f:
            self.original_page = f.readlines()
        self.prpts = Property(self.page_path)
        self.contents = Content(self.page_path)

    def is_publishable_page(self):
        """
        Check if the page is marked as publishable.

        Returns:
        - bool: True if the page has the property `blog-post` set to True, otherwise False.
        """
        return self.prpts.is_blog_post()

    def write_page(self,
                   new_page_path: Path, new_attach_path: Path = None,
                   include_all: bool = False,
                   exist_ok: bool = False, exclude_prpt_keys=None, prefix=None,
                   remove_comments=False):
        """
        Write the page to a new file, including its properties and contents.

        Arguments:
        - new_page_path (Path): The target path to write the page.
        - new_attach_path (Path, optional): Directory to save attachments.
        - include_all (bool): If True, include all properties in the output, regardless of filtering.
        - exist_ok (bool): If False, raise an error if the target file already exists.
        - exclude_prpt_keys (list, optional): A list of property keys to exclude from the output.
        - prefix (str, optional): A string to prefix the file name with. If provided, the resulting
                                  file name will be `prefix_original_name`.
        """
        # Create the target directory for the new page
        new_page_target_dir = new_page_path.parent
        new_page_target_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        # Create the target directory for attachments if provided
        new_attach_target_dir = new_attach_path.parent if new_attach_path else None
        if new_attach_target_dir:
            new_attach_target_dir.mkdir(parents=True, exist_ok=True)

        # Apply prefix to the file name if provided
        if prefix:
            new_page_path = new_page_target_dir / f"{prefix}-{new_page_path.name}"
            new_attach_path = new_attach_target_dir / f"{prefix}-{new_attach_path.name}" if new_attach_path else None

        # Pass exclude_prpt_keys to write_properties
        self.prpts.write_properties(
            new_page_path,
            include_all=include_all, exist_ok=exist_ok, exclude_keys=exclude_prpt_keys
        )
        self.contents.write_contents(
            new_page_path=new_page_path,
            new_attach_path=new_attach_path,
            remove_comments=remove_comments)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=False)
    args = parser.parse_args()

    md_path = Path(args.input)
    Page = ObsidianPage(md_path)
    prpts = Page.prpts.get_prpts()
    attached_files = Page.contents.get_attached()
    contents = Page.contents.get_contents()

    print("Test")
    print("=== Property Check ===")
    for key in prpts:
        print(prpts[key])
    print("=== Content Check :: Property ===")
    for file in attached_files:
        print(file)
    print("=== Content Check :: Content ===")
    for line in contents:
        print(line)

    if (args.output):
        print("=== Write Check ===")
        new_path = Path(args.output)
        Page.write_page(new_path, exist_ok=True)



