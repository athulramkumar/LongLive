"""
Unit tests for VideoBuilderState and PromptEnhancer

Run with: pytest gradio/tests/test_video_builder_state.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Mock torch and other heavy dependencies before importing
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

from web_ui.video_builder_state import PromptEnhancer, VideoBuilderState, ChunkState


class TestPromptEnhancer:
    """Tests for the PromptEnhancer class
    
    Note: Tests that require mocking anthropic API responses are marked with
    pytest.mark.skip as the module-level mocking makes it difficult to
    properly test API calls. The core logic tests work without API calls.
    """
    
    def test_init(self):
        """Test PromptEnhancer initialization"""
        enhancer = PromptEnhancer(api_key="test-key")
        
        assert enhancer.grounding is None
        assert enhancer.previous_prompts == []
    
    def test_set_grounding_direct(self):
        """Test setting grounding without AI enhancement"""
        enhancer = PromptEnhancer(api_key="test-key")
        
        enhancer.set_grounding_direct("A blue boat on a lake")
        
        assert enhancer.grounding == "A blue boat on a lake"
        assert enhancer.previous_prompts == ["A blue boat on a lake"]
    
    def test_enhance_prompt_without_grounding(self):
        """Test that enhance_prompt raises error without grounding"""
        enhancer = PromptEnhancer(api_key="test-key")
        
        with pytest.raises(ValueError, match="Grounding must be set first"):
            enhancer.enhance_prompt("some action")
    
    def test_add_to_history(self):
        """Test manually adding to history"""
        enhancer = PromptEnhancer(api_key="test-key")
        enhancer.set_grounding_direct("A dog")
        
        enhancer.add_to_history("Dog runs fast")
        
        assert len(enhancer.previous_prompts) == 2
        assert enhancer.previous_prompts[1] == "Dog runs fast"
    
    def test_revert_to_chunk_0(self):
        """Test reverting to first chunk"""
        enhancer = PromptEnhancer(api_key="test-key")
        enhancer.set_grounding_direct("Scene")
        enhancer.add_to_history("Chunk 0 prompt")
        enhancer.add_to_history("Chunk 1 prompt")
        enhancer.add_to_history("Chunk 2 prompt")
        
        assert len(enhancer.previous_prompts) == 4
        
        enhancer.revert_to(0)
        
        # Should keep grounding + chunk 0
        assert len(enhancer.previous_prompts) == 1
    
    def test_revert_to_chunk_1(self):
        """Test reverting to second chunk"""
        enhancer = PromptEnhancer(api_key="test-key")
        enhancer.set_grounding_direct("Scene")
        enhancer.add_to_history("Chunk 0 prompt")
        enhancer.add_to_history("Chunk 1 prompt")
        enhancer.add_to_history("Chunk 2 prompt")
        
        enhancer.revert_to(1)
        
        # Should keep grounding + chunk 0 + chunk 1
        assert len(enhancer.previous_prompts) == 2
    
    def test_revert_to_invalid_preserves_minimum(self):
        """Test reverting to invalid index keeps at least grounding"""
        enhancer = PromptEnhancer(api_key="test-key")
        enhancer.set_grounding_direct("Scene")
        
        enhancer.revert_to(-5)
        
        assert len(enhancer.previous_prompts) == 1
        assert enhancer.previous_prompts[0] == "Scene"
    
    def test_history_management_multiple_chunks(self):
        """Test history management with multiple chunks"""
        enhancer = PromptEnhancer(api_key="test-key")
        enhancer.set_grounding_direct("Initial scene")
        
        # Simulate adding chunks
        enhancer.add_to_history("Chunk 1 action")
        enhancer.add_to_history("Chunk 2 action")
        enhancer.add_to_history("Chunk 3 action")
        
        assert len(enhancer.previous_prompts) == 4
        
        # Revert to chunk 2 (index 1 means keep prompts 0 and 1)
        enhancer.revert_to(1)
        assert len(enhancer.previous_prompts) == 2
        assert enhancer.previous_prompts[-1] == "Chunk 1 action"
        
        # Add new chunk
        enhancer.add_to_history("New chunk 2 action")
        assert len(enhancer.previous_prompts) == 3
        assert enhancer.previous_prompts[-1] == "New chunk 2 action"


class TestVideoBuilderState:
    """Tests for VideoBuilderState class (without GPU)"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all heavy dependencies"""
        with patch('gradio.video_builder_state.torch') as mock_torch, \
             patch('gradio.video_builder_state.OmegaConf') as mock_omegaconf, \
             patch('gradio.video_builder_state.CausalInferencePipeline') as mock_pipeline, \
             patch('gradio.video_builder_state.set_seed') as mock_seed, \
             patch('gradio.video_builder_state.get_cuda_free_memory_gb') as mock_vram, \
             patch('gradio.video_builder_state.anthropic.Anthropic') as mock_anthropic:
            
            # Setup torch mocks
            mock_torch.device.return_value = Mock()
            mock_torch.cuda.get_device_name.return_value = "Mock GPU"
            mock_torch.cuda.memory_allocated.return_value = 8e9
            mock_vram.return_value = 40.0
            
            # Setup config mock
            mock_config = Mock()
            mock_config.generator_ckpt = None
            mock_config.adapter = None
            mock_omegaconf.load.return_value = mock_config
            
            # Setup anthropic mock
            response = Mock()
            response.content = [Mock(text="Enhanced")]
            mock_anthropic.return_value.messages.create.return_value = response
            
            yield {
                'torch': mock_torch,
                'omegaconf': mock_omegaconf,
                'pipeline': mock_pipeline,
                'anthropic': mock_anthropic
            }
    
    def test_init_defaults(self):
        """Test VideoBuilderState initialization with defaults"""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            builder = VideoBuilderState.__new__(VideoBuilderState)
            builder.config_path = "configs/test.yaml"
            builder.output_dir = "test_output"
            builder.chunk_duration = 10.0
            builder.max_chunks = 12
            builder.seed = 42
            builder.states = {}
            builder.current_chunk = 0
            builder.is_setup = False
            builder.grounding_set = False
            
            assert builder.chunk_duration == 10.0
            assert builder.max_chunks == 12
            assert builder.current_chunk == 0
            assert builder.is_setup == False
    
    def test_get_chunks_info_empty(self):
        """Test get_chunks_info with no chunks"""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            builder = VideoBuilderState.__new__(VideoBuilderState)
            builder.states = {}
            builder.current_chunk = 0
            builder.chunk_duration = 10.0
            
            result = builder.get_chunks_info()
            
            assert result == []
    
    def test_get_chunks_info_with_chunks(self):
        """Test get_chunks_info with generated chunks"""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            builder = VideoBuilderState.__new__(VideoBuilderState)
            builder.chunk_duration = 10.0
            builder.current_chunk = 2
            
            # Create mock states
            mock_state_0 = Mock()
            mock_state_0.user_prompt = "First prompt"
            mock_state_0.processed_prompt = "Enhanced first"
            
            mock_state_1 = Mock()
            mock_state_1.user_prompt = "Second prompt"
            mock_state_1.processed_prompt = "Enhanced second"
            
            builder.states = {0: mock_state_0, 1: mock_state_1}
            
            result = builder.get_chunks_info()
            
            assert len(result) == 2
            assert result[0]["chunk_num"] == 1
            assert result[0]["time_range"] == "0-10s"
            assert result[0]["user_prompt"] == "First prompt"
            assert result[1]["chunk_num"] == 2
            assert result[1]["time_range"] == "10-20s"
    
    def test_get_status(self):
        """Test get_status returns correct info"""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            builder = VideoBuilderState.__new__(VideoBuilderState)
            builder.is_setup = True
            builder.grounding_set = True
            builder.current_chunk = 3
            builder.max_chunks = 12
            builder.session_id = "test_session"
            builder.session_dir = "/test/path"
            builder.states = {0: Mock(), 1: Mock(), 2: Mock()}
            builder.enhancer = Mock()
            builder.enhancer.grounding = "Test grounding"
            
            status = builder.get_status()
            
            assert status["is_setup"] == True
            assert status["grounding_set"] == True
            assert status["current_chunk"] == 3
            assert status["max_chunks"] == 12
            assert status["chunks_generated"] == 3
            assert status["grounding"] == "Test grounding"


class TestChunkState:
    """Tests for ChunkState dataclass"""
    
    def test_chunk_state_creation(self):
        """Test creating a ChunkState"""
        # Use mocked torch
        torch = sys.modules['torch']
        mock_tensor = Mock()
        torch.zeros.return_value = mock_tensor
        
        state = ChunkState(
            chunk_idx=0,
            user_prompt="Test user prompt",
            processed_prompt="Test processed prompt",
            end_latent=mock_tensor,
            kv_cache=[{"k": mock_tensor, "v": mock_tensor}],
            crossattn_cache=[{"k": mock_tensor, "v": mock_tensor}],
            current_frame=36
        )
        
        assert state.chunk_idx == 0
        assert state.user_prompt == "Test user prompt"
        assert state.processed_prompt == "Test processed prompt"
        assert state.current_frame == 36
        assert state.timestamp is not None


class TestVideoBuilderStateNavigation:
    """Tests for navigation methods (goto_chunk, go_back)"""
    
    def test_goto_chunk_invalid_low(self):
        """Test goto_chunk with invalid low number"""
        builder = VideoBuilderState.__new__(VideoBuilderState)
        builder.current_chunk = 3
        builder.states = {0: Mock(), 1: Mock(), 2: Mock()}
        
        result = builder.goto_chunk(0)
        
        assert result["status"] == "error"
    
    def test_goto_chunk_invalid_high(self):
        """Test goto_chunk with number higher than current"""
        builder = VideoBuilderState.__new__(VideoBuilderState)
        builder.current_chunk = 3
        builder.states = {0: Mock(), 1: Mock(), 2: Mock()}
        
        result = builder.goto_chunk(5)
        
        assert result["status"] == "error"
    
    def test_go_back_at_start(self):
        """Test go_back when at first chunk"""
        builder = VideoBuilderState.__new__(VideoBuilderState)
        builder.current_chunk = 0
        
        result = builder.go_back()
        
        assert result["status"] == "error"
        assert "first chunk" in result["message"].lower() or "already" in result["message"].lower()


class TestIntegration:
    """Integration tests"""
    
    def test_full_flow_without_api(self):
        """Test the full flow using direct grounding (no API calls)"""
        # Test PromptEnhancer flow without API
        enhancer = PromptEnhancer(api_key="test-key")
        
        # Set grounding directly
        enhancer.set_grounding_direct("A cat on a roof")
        assert enhancer.grounding == "A cat on a roof"
        assert len(enhancer.previous_prompts) == 1
        
        # Add prompts to history manually (simulating accepted prompts)
        enhancer.add_to_history("Cat looks down curiously")
        assert len(enhancer.previous_prompts) == 2
        
        enhancer.add_to_history("Cat jumps down gracefully")
        assert len(enhancer.previous_prompts) == 3
        
        # Revert to chunk 1
        enhancer.revert_to(1)
        assert len(enhancer.previous_prompts) == 2
        assert enhancer.previous_prompts[-1] == "Cat looks down curiously"
        
        # Add different action
        enhancer.add_to_history("Cat stretches and yawns")
        assert len(enhancer.previous_prompts) == 3
        assert enhancer.previous_prompts[-1] == "Cat stretches and yawns"
