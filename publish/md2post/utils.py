import logging

logger = logging.getLogger(__name__)

def is_blog_prpt(key):
    """
    Check if a given key is a blog-related property.

    Arguments:
    - key (str): A property key from the YAML frontmatter.

    Returns:
    - bool: True if the key starts with "blog-", otherwise False.
    """
    if key.startswith("blog-"):
        return True
    else:
        return False


def get_prpts_idx(obsidian_page_lines):
    """
    Identify the start and end indices of the YAML frontmatter in an Obsidian markdown file.

    Arguments:
    - obsidian_page_lines (list): A list of strings, where each string is a line from the markdown file.

    Returns:
    - tuple: (prpt_start_idx, prpt_end_idx)
        - prpt_start_idx (int): The index of the line containing the first `---`.
        - prpt_end_idx (int): The index of the line containing the second `---`.

    Logging:
    - Logs a warning if no valid YAML frontmatter is found.

    Behavior:
    - This function searches for lines starting with `---` to identify the boundaries of the YAML frontmatter.
    - Validates that the frontmatter contains at least one key-value pair using a colon `:`.

    Example:
    ```
    ---
    blog-post: true
    blog-title: "Sample Blog Post"
    ---
    Content goes here...
    ```
    """
    prpt_start_idx = 0
    prpt_end_idx = 0
    found_first_separator = False
    found_second_separator = False

    # Locate the start and end of the YAML frontmatter
    for i, line in enumerate(obsidian_page_lines):
        line = line.lstrip()
        line = line.rstrip()
        if line == "---":
            # Found the opening marker
            if not found_first_separator:
                found_first_separator = True
                prpt_start_idx = i
            # Found the closing marker
            elif found_first_separator and not found_second_separator:
                found_second_separator = True
                prpt_end_idx = i

            # Stop searching after finding both markers
            if found_first_separator and found_second_separator:
                break

    # Check if the frontmatter contains key-value pairs
    prpts_lines = obsidian_page_lines[prpt_start_idx+1:prpt_end_idx]
    found_colon = False
    for line in prpts_lines:
        if ":" in line:
            found_colon = True
            break

    if prpt_end_idx == 0 or not found_colon:
        logger.warning("The page may not contain any properties or the format is incorrect.")
        return 0, 0

    return prpt_start_idx, prpt_end_idx

def depth_first_search(path, collect_list, callback_fn=lambda _: True):
    """
    Perform a depth-first search (DFS) on a directory structure to collect files or directories.

    Arguments:
    - path (Path): The starting directory or file path.
    - collect_list (list): A list to store the collected paths that meet the callback function's condition.
    - callback_fn (function): A callback function that takes a `Path` object and returns `True` or `False`.
                              Only paths that return `True` are added to the `collect_list`.

    Behavior:
    - If the path is a file and passes the `callback_fn`, it is added to `collect_list`.
    - If the path is a directory, the function recursively explores its contents.

    Example:
    ```
    # Collect all Markdown files from a directory
    markdown_files = []
    depth_first_search(Path("/my/vault"), markdown_files, callback_fn=is_markdown)
    ```
    """
    if path.is_file() and callback_fn(path):
        collect_list.append(path)
    elif path.is_dir():
        for item in path.iterdir():
            depth_first_search(item, collect_list, callback_fn=callback_fn)