import os
import sys
import unittest

from utils.load_split import load_and_split_md

md_path = r"D:\Download\AlgoNote-main\AlgoNote-main\docs\02_linked_list\02_01_linked_list_basic.md"
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
class TestLoadAndSplitMD(unittest.TestCase):
    def setUp(self):
        """测试前准备"""
        self.test_md_path = "test_document.md"
        # 创建测试文件
        with open(self.test_md_path, 'w', encoding='utf-8') as f:
            f.write("# 测试文档\n\n这是一个测试Markdown文件。\n\n## 第二部分\n\n这是文档的第二部分内容。")

    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.test_md_path):
            os.remove(self.test_md_path)

    def test_load_and_split_md(self):
        """测试Markdown文件加载和分割"""
        try:
            chunks = load_and_split_md(self.test_md_path)

            # 验证结果
            self.assertIsInstance(chunks, list)
            self.assertGreater(len(chunks), 0)

            # 验证每个chunk都有正确的属性
            for chunk in chunks:
                self.assertTrue(hasattr(chunk, 'page_content'))
                self.assertTrue(hasattr(chunk, 'metadata'))
                self.assertIsInstance(chunk.page_content, str)
                self.assertIsInstance(chunk.metadata, dict)

            print(f"测试成功: 生成 {len(chunks)} 个块")

        except Exception as e:
            self.fail(f"测试失败: {e}")

    def test_nonexistent_file(self):
        """测试不存在的文件"""
        with self.assertRaises(FileNotFoundError):
            load_and_split_md("nonexistent.md")

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
