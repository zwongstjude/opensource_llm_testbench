# LLM Testing

This repo is mainly to explore the relationship between vram usage, quantization, model parameters and token generation speed. 

### Running the test
`test_model.py`
to run
```
python test_model.py --model=<model card from HF> --quantization=<4 or 8 else 16> --token=HF_TOKEN
```
after the model runs, it runs a loop that allows users to write in custom prompts

### Writing new prompts
`prompt.json` 
a simple json structure that stores prompts one may want to feed into the llm
make sure the labels field corresponds with the prompts field 


### Reading outputs
`out.txt`
model outputs, the first line is the tokens per second in order 
the next is the memory usage after running through all the prompts 
nvidia-smi shows gpu stats when the when it's idle