from md2post import *

logger = logging.getLogger(__name__)

class Property(object):
    """
    Represents the YAML frontmatter (properties) of an Obsidian page.

    This class is responsible for:
    - Parsing the YAML frontmatter section of a markdown file.
    - Managing key-value pairs in the frontmatter.
    - Writing modified properties back to a markdown file.

    Key Features:
    - Extracts properties from the frontmatter between `---` markers.
    - Supports filtering or excluding specific properties when writing.
    - Validates the structure of the frontmatter and ensures it conforms to YAML syntax.

    Attributes:
    - page_path (Path): Path to the markdown file.
    - prpts_start_idx (int): Index of the start of the frontmatter.
    - prpts_end_idx (int): Index of the end of the frontmatter.
    - prpts_lines (list): List of lines representing the properties.
    - prpts (dict): Dictionary representation of the properties.
    """
    def __init__(self, page_path: Path):
        """
        Initialize the Property instance by parsing the YAML frontmatter.

        Arguments:
        - page_path (Path): Path to the markdown file.

        This method extracts the frontmatter section of the file and parses it into a
        dictionary (`prpts`) for easy access and manipulation.
        """
        logger.debug("Initializing Property with page_path: %s", page_path)
        self.page_path = page_path.resolve()
        with open(self.page_path, "r") as f:
            obsidian_page = f.readlines()
        logger.debug("Read %d lines from the page", len(obsidian_page))

        self.prpts_start_idx, self.prpts_end_idx = get_prpts_idx(obsidian_page)
        logger.debug("Frontmatter indexes found: start=%d, end=%d",
                     self.prpts_start_idx, self.prpts_end_idx)

        self.prpts_lines = obsidian_page[self.prpts_start_idx+1:self.prpts_end_idx]
        logger.debug("Extracted %d lines of properties", len(self.prpts_lines))

        self.prpts = {}
        for prpt_lines in self.__iter_prpts(self.prpts_lines):
            key, val = self.parse_single_prpt(prpt_lines)
            self.prpts[key] = val
        
        logger.info("Parsed %d properties from frontmatter", len(self.prpts))

    def get_prpts(self):
        """
        Retrieve all parsed properties as a dictionary.

        Returns:
        - dict: Key-value pairs representing the properties.
        """
        return self.prpts

    def is_blog_post(self):
        """
        Check if the page is marked as a blog post.

        Returns:
        - bool: True if the `blog-post` property exists and is set to True, otherwise False.
        """
        return self.prpts.get("blog-post", False)

    def write_properties(self, new_page_path: Path, include_all: bool = False,
                         exist_ok: bool = False, exclude_keys=None):
        """
        Write properties to a new file, optionally excluding specific keys.

        Arguments:
        - new_page_path (Path): The target path to write properties.
        - include_all (bool): If True, include all properties regardless of filtering.
        - exist_ok (bool): If False, raises an error if the file already exists.
        - exclude_keys (list): List of keys to exclude from the output.

        This method writes the YAML frontmatter to the target file and ensures
        proper formatting of the key-value pairs.
        """
        if exclude_keys is None:
            exclude_keys = []

        if not exist_ok and new_page_path.exists():
            logger.info("Target path already exists.")
            exit(1)

        with open(new_page_path, "w") as f:
            f.write("---\n")
            for key in self.prpts:
                if key in exclude_keys:  # Skip excluded keys
                    continue
                if not include_all and not is_blog_prpt(key):
                    continue
                vals = self.prpts[key]
                key = key.replace("blog-", "")  # Clean up blog prefix
                f.write(f"{key}: ")
                if len(vals) == 1:
                    f.write(f"{vals[0]}\n")
                else:
                    f.write("\n")
                    for val in vals:
                        f.write(f"  - {val}\n")
            f.write("---\n")

    def parse_single_prpt(self, prpt_lines):
        """
        Parse a single key-value pair or list from the frontmatter.

        Arguments:
        - prpt_lines (list): A list of lines representing a single property.

        Returns:
        - tuple: (key, value) where value is a list of parsed values.

        Raises:
        - AssertionError: If the property has multiple keys.
        """
        # Does it have multiple properties? Raise assertion
        prpt_lines = self.__sanity_check(prpt_lines)

        if len(prpt_lines) == 1:
            line = prpt_lines[0]
            val_list = []

            key, val = line.split(":", 1)
            key = key.rstrip()
            val = val.lstrip()
            val_list.append(val)

            return key, val_list
        else:
            val_list = []
            for line in prpt_lines:
                # If there is a colon in the line, it has a key
                if ":" in line:
                    split_line = line.split(":", 1)
                    key = split_line[0].rstrip()
                # If the line starts with "-", it has a value
                elif line.startswith("-"):
                    split_line = line.split("-", 1)
                    val = split_line[-1].lstrip()
                    val_list.append(val)

            return key, val_list

    def __sanity_check(self, prpt_lines):
        """
        Validate the structure of a single property.

        Arguments:
        - prpt_lines (list): Lines representing a single property.

        Returns:
        - list: Sanitized property lines with leading/trailing whitespace removed.

        Raises:
        - AssertionError: If multiple key-value pairs are found in the same property.
        """
        colon_cnt = 0
        cleaned_prpt_lines = []
        for line in prpt_lines:
            # Skip empty line
            if line.strip() == "":
                continue
            else:
                # Remove whitespaces on each side
                line = line.lstrip()
                line = line.rstrip()
                cleaned_prpt_lines.append(line)
            if ":" in line:
                colon_cnt += 1
        if colon_cnt != 1:
            logger.error("Given property has multiple key-value pairs.")
            exit(1)
        return cleaned_prpt_lines

    def __iter_prpts(self, prpts_lines):
        """
        Iterate over all properties in the frontmatter.

        Arguments:
        - prpts_lines (list): Lines representing the frontmatter.

        Yields:
        - list: Lines representing a single property.
        """
        num_line_prpt_start = 0
        num_line_prpt_nxt_start = 0
        for i, line in enumerate(prpts_lines):
            if ":" in line:
                num_line_prpt_start = i

                # Search for the next colon
                found_nxt_colon = False
                prpt_lines = prpts_lines[num_line_prpt_start+1:]
                for j, line_ in enumerate(prpt_lines):
                    if ":" in line_:
                        num_line_prpt_nxt_start = i + j + 1
                        found_nxt_colon = True
                        break
                if not found_nxt_colon:
                    num_line_prpt_nxt_start = len(prpts_lines) + 1
                prpt_lines = prpts_lines[num_line_prpt_start:num_line_prpt_nxt_start]
                yield prpt_lines