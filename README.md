## Prepare Mining in Bittensor
### Test Your LLM in Local
Ask a locally installed LLM to make sure it provides the correct answer. This process is necessary when tuning the model or prompting it to satisfy the answer conditions required by subnet.


Install essential packages
``` bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes sentencepiece
pip install bittensor
```


Save the script to the directory as below
```
~/mistral_test
├── miner.py
└── mistral_runner.py
```

Execute Mistral
```bash
python miner.py
```

This is the test code for setting detailed prompts and whether LLM is working properly locally. Depending on the netuid you want to mine, you need to modify some of the mistral_runner.py or recreate miner.py to the format.
