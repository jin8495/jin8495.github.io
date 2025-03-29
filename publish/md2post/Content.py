from md2post import *

logger = logging.getLogger(__name__)

class Content(object):
    """
    Represents the content of an Obsidian page, handling both the main text content
    and any attachments linked within the markdown file.

    This class is responsible for:
    - Parsing and extracting the main content lines from the markdown file.
    - Managing attachment files referenced within the content (e.g., images or other media).
    - Writing the parsed content and attachments to a new location.

    Key Features:
    - Extracts attachments referenced with the `![[attachment]]` syntax.
    - Supports updating attachment paths when writing content to a new location.
    - Handles binary copying of attachment files to preserve their integrity.

    Attributes:
    - page_path (Path): Path to the markdown file.
    - dir_path (Path): Directory containing the markdown file.
    - contents_lines (list): Lines of content from the markdown file (excluding frontmatter).
    - attached (list): List of attachments with metadata like file path and binary data.
    - contents (list): Cleaned content lines with updated attachment paths.
    """
    def __init__(self, page_path: Path):
        """
        Initialize the Content instance by reading and parsing the markdown file.

        Arguments:
        - page_path (Path): Path to the markdown file.

        This method reads the markdown file, extracts the main content, identifies
        any attachments, and prepares the content for further processing.
        """
        logger.debug("Initializing Content with page_path: %s", page_path)
        self.page_path = page_path.resolve()
        self.dir_path = self.page_path.parent
        with open(self.page_path, "r") as f:
            obsidian_page = f.readlines()
        logger.debug("Read %d lines from file", len(obsidian_page))

        self.prpts_start_idx, self.prpts_end_idx = get_prpts_idx(obsidian_page)
        logger.debug("Frontmatter indexes found: start=%d, end=%d", self.prpts_start_idx, self.prpts_end_idx)

        self.contents_lines = obsidian_page[self.prpts_end_idx+1:]
        self.attached = []
        for num_line, idx, path, binary in self.__iter_attached_previews(self.contents_lines):
            self.attached.append({
                "num_line": num_line,
                "idx": idx,
                "path": path,
                "bin": binary
            })
        logger.info("Total attachments found: %d", len(self.attached))

        self.contents = self.__get_contents_lines(self.contents_lines)
        logger.debug("Finished initializing Content, %d lines of content processed", len(self.contents))

    def get_attached(self):
        """
        Retrieve the list of attachments referenced in the content.

        Returns:
        - list: A list of dictionaries containing metadata about each attachment,
                including its line number, index in the line, path, and binary data.
        """
        return self.attached

    def get_contents(self):
        """
        Retrieve the cleaned and processed content lines.

        Returns:
        - list: A list of content lines with updated attachment paths.
        """
        return self.contents

    def write_contents(self, new_page_path: Path, remove_comments=False):
        """
        Write the content and its attachments to a new location.

        Arguments:
        - new_page_path (Path): The target file path for the content.
        - remove_comments (bool): If True, remove lines between # blog-comments-start and # blog-comments-end.

        This method writes the processed content lines to the specified path and
        copies the referenced attachments to the appropriate location.
        """
        logger.debug("Writing contents to %s, remove_comments=%s", new_page_path, remove_comments)
        # Write content page
        with open(new_page_path, "a") as f:
            contents = self.__fill_attachements(self.contents, self.attached)

            # Remove comment blocks if the flag is True
            if remove_comments:
                contents = self.__remove_comment_blocks(contents)

            for content_line in contents:
                f.write(content_line)

        # Copy attachments
        new_dir_path = new_page_path.parent
        for attachment in self.attached:
            old_attachment_path = attachment["path"]
            old_attachment_name = old_attachment_path.name

            # mkdir
            old_attachment_dir = old_attachment_path.parent
            new_attachment_dir = new_dir_path / old_attachment_dir
            new_attachment_path = new_attachment_dir / old_attachment_name
            new_attachment_dir.mkdir(exist_ok=True)

            # Write attachment binary
            with open(new_attachment_path, "wb") as f:
                f.write(attachment["bin"])
        logger.info("Contents and attachments written to %s", new_page_path)

    def __fill_attachements(self, contents, attached):
        """
        Replace placeholders in content with actual paths to attachments.

        Arguments:
        - contents (list): The original content lines.
        - attached (list): List of attachments with their metadata.

        Returns:
        - list: Updated content lines with actual attachment paths inserted.
        """
        filled_contents = contents
        for attachement in attached:
            filled_contents = self.__fill_single_attachement(filled_contents, attachement)
        return filled_contents

    def __fill_single_attachement(self, contents, attachement):
        """
        Insert the actual attachment path into a specific placeholder.

        Arguments:
        - contents (list): The content lines.
        - attachement (dict): Metadata for a specific attachment.

        Returns:
        - list: Content lines with the specified attachment path updated.
        """
        num_line = attachement["num_line"]
        idx = attachement["idx"]
        path = attachement["path"]

        # Get an exact position to put path
        contents_line = contents[num_line]
        target_idx = 0
        for i in range(idx+1):
            idx_left_parenthesis = contents_line.find("![[")
            contents_line = contents_line[idx_left_parenthesis+3:]
            target_idx += idx_left_parenthesis+3

        # Put path
        contents_line = contents[num_line]
        new_contents_line = contents_line[:target_idx] + str(path) + contents_line[target_idx:]
        contents[num_line] = new_contents_line

        return contents

    def __iter_attached_previews(self, contents_lines):
        """
        Iterate over all attachment placeholders in the content lines.

        Arguments:
        - contents_lines (list): The lines of content to search for attachments.

        Yields:
        - tuple: A tuple containing:
            - num_line (int): Line number in the content.
            - idx (int): Index of the attachment in the line.
            - path (Path): Path to the attachment file.
            - binary (bytes): Binary data of the attachment file.
        """
        for num_line, line in enumerate(contents_lines):
            idx = 0
            while True:
                idx_left_parenthesis = line.find("![[")
                idx_right_parenthesis = line.find("]]")
                if idx_left_parenthesis == -1:
                    break

                attached_file_path = line[idx_left_parenthesis+3:idx_right_parenthesis]
                # Remove the pipe character and any text after it if present
                if "|" in attached_file_path:
                    attached_file_path = attached_file_path.split("|")[0]
                attached_file_path = Path(attached_file_path)

                with open(self.dir_path/attached_file_path, "rb") as f:
                    binary = f.read()
                yield num_line, idx, attached_file_path, binary

                line = line[idx_right_parenthesis+1:]
                idx_left_parenthesis = 0
                idx_right_parenthesis = 0
                idx += 1 

    def __get_contents_lines(self, contents_lines):
        """
        Process content lines to remove placeholders for attachments.

        Arguments:
        - contents_lines (list): The original content lines.

        Returns:
        - list: Processed content lines with placeholders removed.
        """
        removed_contents_lines = []
        for contents_line in contents_lines:
            removed_contents_line = ""
            line = contents_line
            while True:
                idx_left_parenthesis = line.find("![[")
                if idx_left_parenthesis == -1:
                    removed_contents_line += line
                    break

                # Remove attached file path -- append left side of the parenthesis
                removed_contents_line += line[:idx_left_parenthesis+3]

                line = line[2:]
                idx_right_parenthesis = line.find("]]")
                line = line[idx_right_parenthesis:]

            removed_contents_lines.append(removed_contents_line)
        return removed_contents_lines

    def __remove_comment_blocks(self, contents):
        """
        Remove lines between # blog-comments-start and # blog-comments-end.

        Arguments:
        - contents (list): The content lines.

        Returns:
        - list: Content lines with the comment blocks removed.
        """
        cleaned_contents = []
        in_comment_block = False

        for line in contents:
            # Check for comment block start
            if line.strip() == "# blog-comments-start":
                in_comment_block = True
                continue

            # Check for comment block end
            if line.strip() == "# blog-comments-end":
                in_comment_block = False
                continue

            # Skip lines inside a comment block
            if in_comment_block:
                continue

            # Add line if not in a comment block
            cleaned_contents.append(line)

        return cleaned_contents