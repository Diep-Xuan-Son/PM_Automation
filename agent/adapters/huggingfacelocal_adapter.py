import asyncio
import functools
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, AsyncIterator, Dict, List, Any, Optional
from litellm.types.utils import GenericStreamingChunk, ModelResponse
from litellm.llms.custom_llm import CustomLLM
from litellm import completion, acompletion

# Try to import HuggingFace dependencies
try:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class HuggingFaceLocalAdapter(CustomLLM):
    """HuggingFace Local Adapter for running vision-language models locally."""
    
    def __init__(self, device: str = "auto", **kwargs):
        """Initialize the adapter.
        
        Args:
            device: Device to load model on ("auto", "cuda", "cpu", etc.)
            **kwargs: Additional arguments
        """
        super().__init__()
        self.device = device
        self.models = {}  # Cache for loaded models
        self.processors = {}  # Cache for loaded processors
        self._executor = ThreadPoolExecutor(max_workers=1)  # Single thread pool
        
    def _load_model_and_processor(self, model_name: str):
        """Load model and processor if not already cached.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Tuple of (model, processor)
        """
        if model_name not in self.models:
            # Load model
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                attn_implementation="sdpa"
            )
            
            # Load processor
            processor = AutoProcessor.from_pretrained(
                model_name,
                min_pixels=3136,
                max_pixels=4096 * 2160,
                device_map=self.device
            )
            
            # Cache them
            self.models[model_name] = model
            self.processors[model_name] = processor
            
        return self.models[model_name], self.processors[model_name]
    
    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI format messages to HuggingFace format.
        
        Args:
            messages: Messages in OpenAI format
            
        Returns:
            Messages in HuggingFace format
        """
        converted_messages = []
        
        for message in messages:
            converted_message = {
                "role": message["role"],
                "content": []
            }
            
            content = message.get("content", [])
            if isinstance(content, str):
                # Simple text content
                converted_message["content"].append({
                    "type": "text",
                    "text": content
                })
            elif isinstance(content, list):
                # Multi-modal content
                for item in content:
                    if item.get("type") == "text":
                        converted_message["content"].append({
                            "type": "text",
                            "text": item.get("text", "")
                        })
                    elif item.get("type") == "image_url":
                        # Convert image_url format to image format
                        image_url = item.get("image_url", {}).get("url", "")
                        converted_message["content"].append({
                            "type": "image",
                            "image": image_url
                        })
            
            converted_messages.append(converted_message)
            
        return converted_messages
    
    def _generate(self, **kwargs) -> str:
        """Generate response using the local HuggingFace model.
        
        Args:
            **kwargs: Keyword arguments containing messages and model info
            
        Returns:
            Generated text response
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "HuggingFace transformers dependencies not found. "
                "Please install with: pip install \"cua-agent[uitars-hf]\""
            )
        
        # Extract messages and model from kwargs
        messages = kwargs.get('messages', [])
        model_name = kwargs.get('model', 'ByteDance-Seed/UI-TARS-1.5-7B')
        max_new_tokens = kwargs.get('max_tokens', 128)
        
        # Warn about ignored kwargs
        ignored_kwargs = set(kwargs.keys()) - {'messages', 'model', 'max_tokens'}
        if ignored_kwargs:
            warnings.warn(f"Ignoring unsupported kwargs: {ignored_kwargs}")
        
        # Load model and processor
        model, processor = self._load_model_and_processor(model_name)
        
        # Convert messages to HuggingFace format
        hf_messages = self._convert_messages(messages)
        
        # Apply chat template and tokenize
        inputs = processor.apply_chat_template(
            hf_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Move inputs to the same device as model
        inputs = inputs.to(model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            
        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode output
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        return output_text[0] if output_text else ""
    
    def completion(self, *args, **kwargs) -> ModelResponse:
        """Synchronous completion method.
        
        Returns:
            ModelResponse with generated text
        """
        generated_text = self._generate(**kwargs)
        
        return completion(
            model=f"huggingface-local/{kwargs['model']}",
            mock_response=generated_text,
        )
    
    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        """Asynchronous completion method.
        
        Returns:
            ModelResponse with generated text
        """
        # Run _generate in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        generated_text = await loop.run_in_executor(
            self._executor, 
            functools.partial(self._generate, **kwargs)
        )
        
        return await acompletion(
            model=f"huggingface-local/{kwargs['model']}",
            mock_response=generated_text,
        )
    
    def streaming(self, *args, **kwargs) -> Iterator[GenericStreamingChunk]:
        """Synchronous streaming method.
        
        Returns:
            Iterator of GenericStreamingChunk
        """
        generated_text = self._generate(**kwargs)
        
        generic_streaming_chunk: GenericStreamingChunk = {
            "finish_reason": "stop",
            "index": 0,
            "is_finished": True,
            "text": generated_text,
            "tool_use": None,
            "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        }
        
        yield generic_streaming_chunk
    
    async def astreaming(self, *args, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        """Asynchronous streaming method.
        
        Returns:
            AsyncIterator of GenericStreamingChunk
        """
        # Run _generate in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        generated_text = await loop.run_in_executor(
            self._executor, 
            functools.partial(self._generate, **kwargs)
        )
        
        generic_streaming_chunk: GenericStreamingChunk = {
            "finish_reason": "stop",
            "index": 0,
            "is_finished": True,
            "text": generated_text,
            "tool_use": None,
            "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        }
        
        yield generic_streaming_chunk