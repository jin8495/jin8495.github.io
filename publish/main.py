from pathlib import Path
from md2post import *

import argparse

def main(input_path, output_path):
    """
    Main function to process markdown files in an Obsidian Vault and export blog posts.

    Arguments:
    - input_path (Path): Path to the root directory of the Obsidian Vault.
    - output_path (Path): Path to the directory where processed files will be stored.

    Behavior:
    - Searches for all markdown files in the input directory.
    - Filters and processes files marked as publishable (blog-post: true).
    - Copies each blog post and its attachments to the specified output directory, 
      respecting the directory hierarchy defined by the 'blog-directory' property.
    """
    # Collect all markdown files from the input path
    def custom_callback_fn(path: Path):
        """
        Custom callback function to filter paths.
        - Includes only markdown files (.md).
        - Excludes files or directories under paths matching '*_Template'.
        
        Arguments:
        - path (Path): The current path being checked.

        Returns:
        - bool: True if the path should be included, False otherwise.
        """
        # Check if the file is a markdown file
        if path.is_file() and path.suffix != ".md":
            return False

        # Exclude paths under directories matching '*_Template'
        if any(part.endswith("_Templates") for part in path.parts):
            return False

        return True

    markdown_files = []
    depth_first_search(input_path, markdown_files, callback_fn=custom_callback_fn)

    blog_posts = []
    for file_path in markdown_files:
        # Create an ObsidianPage object for each markdown file
        page = ObsidianPage(file_path)
        if page.is_publishable_page():  # Check if the page is marked as publishable
            blog_posts.append(page)

    for blog_post in blog_posts:
        # Retrieve all properties from the blog post
        prpts = blog_post.prpts.get_prpts()
        blog_dir = Path(prpts.get("blog-directory", [""])[0])  # Get the 'blog-directory' property

        if str(blog_dir) == "":
            # Raise an error if 'blog-directory' is missing or empty
            raise ValueError(f"[WARNING] Skipping {blog_post.page_path}: Missing 'blog-directory'.")

        # Extract blog-date for prefix
        blog_date = prpts.get("blog-date", [""][0])
        if blog_date == "":
            raise ValueError(f"[ERROR] Missing 'blog-date' for {blog_post.page_path}")

        # Construct the output path for the blog post
        target_path = output_path / blog_dir
        target_file_path = target_path / blog_post.page_path.name

        # Write the blog post and attachments to the output directory with a prefix
        # Exclude 'blog-post' and 'blog-directory' properties from the output
        blog_post.write_page(
            target_file_path,
            exclude_prpt_keys=['blog-post', 'blog-directory'],
            prefix=blog_date,  # Use blog-date as the prefix,
            remove_comments=True # Remove blog comments if specified
        )

        # Log the successfully copied blog post
        print(f"Copied blog post to: {target_file_path}")

if __name__ == "__main__":
    # Parse command-line arguments for input and output paths
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str, help="Path to the Obsidian Vault.")
    parser.add_argument("-o", "--output", required=True, type=str, help="Path to directory where the result will be stored.")
    args = parser.parse_args()

    # Convert input and output paths to Path objects
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Execute the main function with the provided paths
    main(input_path, output_path)