from piper import PiperVoice
import inspect

# Load a voice
voice = PiperVoice.load("./piper_models/en_US-arctic-medium.onnx", use_cuda=False)

# Check synthesize method signature
print("synthesize() parameters:")
print(inspect.signature(voice.synthesize))
print()

# Check config options
print("Voice config attributes:")
print(vars(voice.config))