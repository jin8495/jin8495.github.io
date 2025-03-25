import unittest
from pathlib import Path
from md2post.Property import Property

class TestProperty(unittest.TestCase):
    def setUp(self):
        # Create a temporary markdown file with frontmatter for testing
        self.temp_file = Path("temp_property.md")
        self.temp_file.write_text(
            "---\n"
            "blog-post: true\n"
            "blog-title: Test Property\n"
            "---\n"
            "Content here.\n"
        )

    def tearDown(self):
        # Remove the temporary file after each test
        if self.temp_file.exists():
            self.temp_file.unlink()

    def test_get_prpts(self):
        # Initialize Property object
        props = Property(self.temp_file)

        # Verify properties
        prpts = props.get_prpts()
        self.assertEqual(prpts["blog-post"], ["true"])
        self.assertEqual(prpts["blog-title"], ["Test Property"])

    def test_is_blog_post(self):
        # Initialize Property object
        props = Property(self.temp_file)
        self.assertTrue(props.is_blog_post())

if __name__ == "__main__":
    unittest.main()