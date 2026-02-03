"""
Integration tests for Gradio web demo

Run with: pytest web_ui/tests/test_web_demo.py -v

Note: These tests mock the gradio module since heavy dependencies may not be available.
The main logic tests are in test_video_builder_state.py.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Mock heavy dependencies FIRST
sys.modules['torch'] = MagicMock()
sys.modules['omegaconf'] = MagicMock()
sys.modules['einops'] = MagicMock()
sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.io'] = MagicMock()
sys.modules['pipeline'] = MagicMock()
sys.modules['pipeline.causal_inference'] = MagicMock()
sys.modules['utils'] = MagicMock()
sys.modules['utils.misc'] = MagicMock()
sys.modules['utils.memory'] = MagicMock()
sys.modules['anthropic'] = MagicMock()

# Create comprehensive gradio mock
mock_gr = MagicMock()
mock_gr.Progress = MagicMock(return_value=MagicMock())
mock_gr.Blocks = MagicMock()
mock_gr.Markdown = MagicMock()
mock_gr.Row = MagicMock()
mock_gr.Column = MagicMock()
mock_gr.Group = MagicMock()
mock_gr.Button = MagicMock()
mock_gr.Textbox = MagicMock()
mock_gr.Checkbox = MagicMock()
mock_gr.Video = MagicMock()
mock_gr.Dataframe = MagicMock()
mock_gr.Number = MagicMock()
mock_gr.File = MagicMock()
mock_gr.update = MagicMock(return_value={})
mock_gr.themes = MagicMock()
mock_gr.themes.Soft = MagicMock(return_value=MagicMock())
sys.modules['gradio'] = mock_gr

# Now import web_demo
from web_ui import web_demo


class TestWebDemoFunctions:
    """Tests for web_demo.py functions"""
    
    @pytest.fixture
    def mock_builder(self):
        """Create a mock VideoBuilderState"""
        mock = Mock()
        mock.is_setup = True
        mock.grounding_set = True
        mock.session_id = "test_session"
        mock.session_dir = "/test/session/path"
        mock.current_chunk = 0
        mock.chunk_duration = 10.0
        mock.enhancer = Mock()
        mock.enhancer.grounding = "Test grounding"
        return mock
    
    @pytest.fixture(autouse=True)
    def reset_builder(self):
        """Reset builder before each test"""
        web_demo.builder = None
        yield
        web_demo.builder = None
    
    def test_enhance_grounding_no_builder(self):
        """Test enhance_grounding when builder is None"""
        web_demo.builder = None
        result = web_demo.enhance_grounding("test prompt", False)
        assert "Error" in result[0]
    
    def test_enhance_grounding_not_setup(self, mock_builder):
        """Test enhance_grounding when model not setup"""
        mock_builder.is_setup = False
        web_demo.builder = mock_builder
        
        result = web_demo.enhance_grounding("test prompt", False)
        assert "Error" in result[0]
    
    def test_enhance_grounding_empty_input(self, mock_builder):
        """Test enhance_grounding with empty input"""
        web_demo.builder = mock_builder
        result = web_demo.enhance_grounding("", False)
        assert "Please enter" in result[0]
    
    def test_enhance_grounding_success(self, mock_builder):
        """Test successful grounding enhancement"""
        mock_builder.set_grounding.return_value = {
            "user_input": "A cat",
            "enhanced": "A fluffy orange cat",
            "skip_ai": False
        }
        web_demo.builder = mock_builder
        
        result = web_demo.enhance_grounding("A cat", False)
        
        assert "enhanced" in result[0].lower() or "review" in result[0].lower()
        assert result[1] == "A fluffy orange cat"
    
    def test_enhance_grounding_skip_ai(self, mock_builder):
        """Test grounding with skip_ai=True"""
        mock_builder.set_grounding.return_value = {
            "user_input": "A dog",
            "enhanced": "A dog",
            "skip_ai": True
        }
        web_demo.builder = mock_builder
        
        web_demo.enhance_grounding("A dog", True)
        mock_builder.set_grounding.assert_called_with("A dog", skip_ai=True)
    
    def test_accept_grounding(self, mock_builder):
        """Test accepting grounding"""
        web_demo.builder = mock_builder
        
        result = web_demo.accept_grounding("Enhanced grounding text")
        
        mock_builder.accept_grounding.assert_called_with("Enhanced grounding text")
        assert len(result) == 4
    
    def test_enhance_chunk_prompt_no_grounding(self, mock_builder):
        """Test enhance_chunk_prompt without grounding set"""
        mock_builder.grounding_set = False
        web_demo.builder = mock_builder
        
        result = web_demo.enhance_chunk_prompt("test action", False)
        assert "Error" in result[0]
    
    def test_enhance_chunk_prompt_empty(self, mock_builder):
        """Test enhance_chunk_prompt with empty input"""
        web_demo.builder = mock_builder
        result = web_demo.enhance_chunk_prompt("", False)
        assert "Please enter" in result[0]
    
    def test_enhance_chunk_prompt_skip_ai(self, mock_builder):
        """Test enhance_chunk_prompt with skip_ai"""
        web_demo.builder = mock_builder
        
        result = web_demo.enhance_chunk_prompt("it moves", True)
        
        assert "no AI" in result[0].lower() or "directly" in result[0].lower()
        assert result[1] == "it moves"
    
    def test_enhance_chunk_prompt_success(self, mock_builder):
        """Test successful chunk prompt enhancement"""
        mock_builder.enhance_chunk_prompt.return_value = "Enhanced action description"
        web_demo.builder = mock_builder
        
        result = web_demo.enhance_chunk_prompt("it jumps", False)
        assert result[1] == "Enhanced action description"
    
    def test_go_back_no_builder(self):
        """Test go_back when builder is None"""
        web_demo.builder = None
        result = web_demo.go_back_handler()
        assert "Error" in result[0]
    
    def test_go_back_success(self, mock_builder):
        """Test successful go_back"""
        mock_builder.go_back.return_value = {"status": "success", "message": "Rewound"}
        mock_builder.get_chunks_info.return_value = []
        mock_builder.current_chunk = 0
        web_demo.builder = mock_builder
        
        web_demo.go_back_handler()
        mock_builder.go_back.assert_called_once()
    
    def test_goto_chunk_invalid(self, mock_builder):
        """Test goto_chunk with invalid input"""
        mock_builder.get_chunks_info.return_value = []  # Return empty list for history
        web_demo.builder = mock_builder
        result = web_demo.goto_chunk_handler(0)
        assert "valid" in result[0].lower() or "enter" in result[0].lower()
    
    def test_goto_chunk_success(self, mock_builder):
        """Test successful goto_chunk"""
        mock_builder.goto_chunk.return_value = {"status": "success", "message": "Rewound to chunk 2"}
        mock_builder.get_chunks_info.return_value = [
            {"chunk_num": 1, "time_range": "0-10s", "user_prompt": "test"}
        ]
        mock_builder.current_chunk = 1
        web_demo.builder = mock_builder
        
        web_demo.goto_chunk_handler(2)
        mock_builder.goto_chunk.assert_called_with(2)
    
    def test_finalize_no_builder(self):
        """Test finalize when builder is None"""
        web_demo.builder = None
        result = web_demo.finalize_video()
        assert "Error" in result[0]
    
    def test_finalize_success(self, mock_builder):
        """Test successful finalize"""
        mock_builder.finalize.return_value = {
            "status": "success",
            "final_video": "/path/to/final.mp4",
            "session_dir": "/path/to/session",
            "total_chunks": 3,
            "duration": "~30s"
        }
        web_demo.builder = mock_builder
        
        result = web_demo.finalize_video()
        
        assert "finalized" in result[0].lower()
        assert result[1] == "/path/to/final.mp4"
    
    def test_get_history_data_no_builder(self):
        """Test get_history_data when builder is None"""
        web_demo.builder = None
        result = web_demo.get_history_data()
        assert result == []
    
    def test_get_history_data_with_chunks(self, mock_builder):
        """Test get_history_data with chunks"""
        mock_builder.get_chunks_info.return_value = [
            {"chunk_num": 1, "time_range": "0-10s", "user_prompt": "First prompt here"},
            {"chunk_num": 2, "time_range": "10-20s", "user_prompt": "Second prompt"},
        ]
        web_demo.builder = mock_builder
        
        result = web_demo.get_history_data()
        
        assert len(result) == 2
        assert result[0][0] == 1
        assert result[0][1] == "0-10s"


class TestCreateDemo:
    """Tests for the create_demo function"""
    
    def test_create_demo_is_callable(self):
        """Test that create_demo is a callable function"""
        assert callable(web_demo.create_demo)


class TestErrorHandling:
    """Tests for error handling"""
    
    @pytest.fixture(autouse=True)
    def reset_builder(self):
        """Reset builder before each test"""
        web_demo.builder = None
        yield
        web_demo.builder = None
    
    def test_enhance_grounding_exception(self):
        """Test enhance_grounding handles exceptions"""
        mock_builder = Mock()
        mock_builder.is_setup = True
        mock_builder.set_grounding.side_effect = Exception("Test error")
        web_demo.builder = mock_builder
        
        result = web_demo.enhance_grounding("test", False)
        assert "Error" in result[0] or "error" in result[0].lower()
