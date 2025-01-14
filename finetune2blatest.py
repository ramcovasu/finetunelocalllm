from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments, TextStreamer
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
import pandas as pd
from faker import Faker
import random
import json
from datetime import datetime
import subprocess
import os
import datasets
from datasets import Dataset

# Initialize Faker with Brazilian Portuguese locale
fake = Faker(['pt_BR'])

class LegalDocumentProcessor:
    def __init__(self):
        self.departments = ["SPECIALIZED CYBER CRIME UNIT", "MINISTRY OF LABOR POLICE", "CIVIL POLICE", "CRIMINAL INVESTIGATION DEPARTMENT"]
        self.laws = [
            "Law No. 129, VI, sd Republic",
            "Law No. 12.830/2013",
            "Law Complementary No. 75/93",
            "Resolution No. 59/2008"
        ]
        self.telcos = ["TIM", "VIVO", "CLARO", "OI"]
        self.states = [
            "São Paulo", "Rio de Janeiro", "Minas Gerais", "Paraná", 
            "Santa Catarina", "Rio Grande do Sul", "Bahia", "Pernambuco"
        ]
        self.building_types = [
            "Torre", "Edifício", "Centro Empresarial", "Complexo", 
            "Business Center", "Corporate Tower"
        ]
        
    def generate_address(self, include_building_details=False):
        """Generate a realistic Brazilian address with optional building details"""
        recipient = fake.company()
        street_name = fake.street_name()
        number = str(random.randint(1, 3000))
        city = fake.city()
        state = random.choice(self.states)
        zip_code = fake.postcode()
        
        address_info = {
            "recipient_name": recipient,
            "street": f"{street_name}, {number}",
            "city": city,
            "state": state,
            "country": "Brazil",
            "zip_code": zip_code
        }
        
        if include_building_details:
            building_name = f"{random.choice(self.building_types)} {fake.last_name()}"
            floor = str(random.randint(1, 30))
            suite = str(random.randint(1, 50))
            address_info["building_details"] = f"{building_name}, Floor {floor}, Suite {suite}"
            address_info["street"] = f"{street_name}, {number}"
            
        return address_info

    def generate_noise_content(self):
        """Generate random noise content for documents"""
        return {
            "reference": f"REF-{random.randint(1000,9999)}/{random.randint(2023,2024)}",
            "phone": f"+55 {random.randint(11,99)} {random.randint(10000,99999)}-{random.randint(1000,9999)}",
            "internal_code": f"INT-{random.randint(100,999)}-{random.choice('ABCDEF')}",
            "priority": random.choice(["URGENT", "NORMAL", "HIGH PRIORITY", "CONFIDENTIAL"]),
            "contact_person": fake.name(),
            "department_code": f"DEPT-{random.randint(1,99):02d}"
        }

    def format_template_1(self, address_info, noise):
        """Standard format - Clean and straightforward"""
        pt_text = f"""À
{address_info['recipient_name']}
{address_info['street']}
{address_info['city']}, {address_info['state']}
{address_info['zip_code']}
{address_info['country']}

NOTIFICAÇÃO de REITERAÇÃO

O(MINISTÉRIO PÚBLICO DO TRABALHO) nos termos do {random.choice(self.laws)},
com vistas à instrução do procedimento em referência, REQUISITA de vissa Se Foria, em
reiteração à Notificação n.º {noise['reference']}, no prazo de 15 (quinze) dias."""

        en_text = f"""To
{address_info['recipient_name']}
{address_info['street']}
{address_info['city']}, {address_info['state']}
{address_info['zip_code']}
{address_info['country']}

OFFICIAL NOTICE of CADASTRAL REPEAT

THE MINISTRY OF LABOR POLICE, pursuant to {random.choice(self.laws)},
with the purpose of instructing the procedure in reference, REQUESTS for review
Notice No. {noise['reference']}, within 15 (fifteen) days."""

        return pt_text, en_text

    def format_template_2(self, address_info, noise):
        """Complex format with building details and additional information"""
        building_details = address_info.get('building_details', '')
        
        pt_text = f"""URGENTE - CONFIDENCIAL
Referência: {noise['reference']}
Contato: {noise['phone']}

DESTINATÁRIO:
{address_info['recipient_name']}
{building_details}
{address_info['street']}
{address_info['city']}, {address_info['state']}
{address_info['zip_code']}
{address_info['country']}

Responsável: {noise['contact_person']}
Código Interno: {noise['internal_code']}

NOTIFICAÇÃO de REITERAÇÃO"""

        en_text = f"""URGENT - CONFIDENTIAL
Reference: {noise['reference']}
Contact: {noise['phone']}

RECIPIENT:
{address_info['recipient_name']}
{building_details}
{address_info['street']}
{address_info['city']}, {address_info['state']}
{address_info['zip_code']}
{address_info['country']}

Responsible: {noise['contact_person']}
Internal Code: {noise['internal_code']}

OFFICIAL NOTICE of CADASTRAL REPEAT"""

        return pt_text, en_text

    def format_template_3(self, address_info, noise):
        """Format with address at the bottom and header information"""
        pt_text = f"""MINISTÉRIO PÚBLICO DO TRABALHO
Departamento: {noise['department_code']}
Prioridade: {noise['priority']}

NOTIFICAÇÃO OFICIAL
Processo No: {noise['reference']}

[CONTEÚDO PRINCIPAL DO DOCUMENTO]

Informações de Entrega:
{address_info['recipient_name']}
{address_info['street']}
{address_info['city']}, {address_info['state']}
{address_info['zip_code']}
{address_info['country']}"""

        en_text = f"""MINISTRY OF LABOR
Department: {noise['department_code']}
Priority: {noise['priority']}

OFFICIAL NOTIFICATION
Process No: {noise['reference']}

[MAIN DOCUMENT CONTENT]

Delivery Information:
{address_info['recipient_name']}
{address_info['street']}
{address_info['city']}, {address_info['state']}
{address_info['zip_code']}
{address_info['country']}"""

        return pt_text, en_text

    def format_template_4(self, address_info, noise):
        """Format with table-like structure and multiple sections"""
        pt_text = f"""DOCUMENTO OFICIAL
═══════════════════════
│ Ref: {noise['reference']}
│ Dept: {noise['department_code']}
═══════════════════════

DESTINATÁRIO
───────────────────────
Empresa: {address_info['recipient_name']}
Endereço: {address_info['street']}
Cidade: {address_info['city']}
Estado: {address_info['state']}
CEP: {address_info['zip_code']}
País: {address_info['country']}

CONTATO
───────────────────────
Tel: {noise['phone']}
Responsável: {noise['contact_person']}"""

        en_text = f"""OFFICIAL DOCUMENT
═══════════════════════
│ Ref: {noise['reference']}
│ Dept: {noise['department_code']}
═══════════════════════

RECIPIENT
───────────────────────
Company: {address_info['recipient_name']}
Address: {address_info['street']}
City: {address_info['city']}
State: {address_info['state']}
ZIP: {address_info['zip_code']}
Country: {address_info['country']}

CONTACT
───────────────────────
Phone: {noise['phone']}
Responsible: {noise['contact_person']}"""

        return pt_text, en_text

    def format_template_5(self, address_info, noise):
        """Format with minimal structure and mixed information"""
        pt_text = f"""Ref: {noise['reference']} | {noise['priority']}
Contato: {noise['phone']}

Para processamento imediato:
{address_info['recipient_name']}
A/C: {noise['contact_person']}
{address_info['street']}
{address_info['city']}, {address_info['state']}
{address_info['zip_code']}
{address_info['country']}

Código: {noise['internal_code']}
Departamento: {noise['department_code']}"""

        en_text = f"""Ref: {noise['reference']} | {noise['priority']}
Contact: {noise['phone']}

For immediate processing:
{address_info['recipient_name']}
Attn: {noise['contact_person']}
{address_info['street']}
{address_info['city']}, {address_info['state']}
{address_info['zip_code']}
{address_info['country']}

Code: {noise['internal_code']}
Department: {noise['department_code']}"""

        return pt_text, en_text

    def generate_document(self):
        """Generate a document using a random template"""
        # Randomly decide whether to include building details
        include_building_details = random.choice([True, False])
        address_info = self.generate_address(include_building_details)
        noise = self.generate_noise_content()
        
        # Select random template
        template_func = random.choice([
            self.format_template_1,
            self.format_template_2,
            self.format_template_3,
            self.format_template_4,
            self.format_template_5
        ])
        
        pt_text, en_text = template_func(address_info, noise)
        return pt_text, en_text, address_info

    def generate_dataset(self, num_samples: int) -> dict:
        instructions = []
        inputs = []
        outputs = []
        
        system_prompt = """You need to be 100% accurate. You are an expert in extracting address information from documents. Extract and return a JSON object with the following fields:
- recipient_name: The full name of the recipient
- street: The street address
- city: The city name
- state: The state name
- country: The country name
- zip_code: The postal/ZIP code

CRITICAL: 
1. Return a complete, valid JSON object
2. Use null for any missing information
3. Translate all address components to English if they are in another language
4. Keep string values clean and simple
5. Be 100% accurate in extraction
6. Include building/floor/suite information as part of the street address if present"""

        user_prompt = """Extract the complete address information from the following document and return it as a JSON object. Be 100% accurate.

Translated Text: {translated_text}

Original Text: {original_text}"""
        
        for _ in range(num_samples):
            pt_text, en_text, address_info = self.generate_document()
            instructions.append(system_prompt)
            inputs.append(user_prompt.format(
                translated_text=en_text,
                original_text=pt_text
            ))
            outputs.append(json.dumps(address_info, indent=2))
            
        return {
            "instruction": instructions,
            "input": inputs,
            "output": outputs
        }



def main():
    # Initialize processor and generate dataset
    print("\nGenerating dataset...")
    processor = LegalDocumentProcessor()
    raw_dataset = processor.generate_dataset(1000)  # Generate 1000 samples

    # Print sample for verification
    print("\nSample data:")
    print("\nInstruction:")
    print(raw_dataset['instruction'][0])
    print("\nInput:")
    print(raw_dataset['input'][0])
    print("\nOutput:")
    print(raw_dataset['output'][0])

    # Model Configuration
    print("\nInitializing model...")
    max_seq_length = 4096
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-2-2b-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Setup LoRA
    print("\nSetting up LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Convert to HuggingFace dataset
    dataset = Dataset.from_dict(raw_dataset)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=3407)
    
    def format_prompt(example):
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}{tokenizer.eos_token}"""
        return {"text": prompt}

    # Format datasets
    formatted_train_dataset = split_dataset['train'].map(format_prompt)
    formatted_eval_dataset = split_dataset['test'].map(format_prompt)

    print(f"\nTrain dataset size: {len(formatted_train_dataset)}")
    print(f"Validation dataset size: {len(formatted_eval_dataset)}")

    # Configure trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_train_dataset,
        eval_dataset=formatted_eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,
            warmup_ratio=0.1,
            max_steps=200,
            learning_rate=5e-5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir="outputs",
            report_to="none",
            save_strategy="steps",
            save_steps=50,
            evaluation_strategy="steps",
            eval_steps=50,
        ),
    )

    print("\nStarting training...")
    trainer_stats = trainer.train()
    print("\nTraining stats:", trainer_stats)

    print("\nSaving model...")
    model.save_pretrained_merged("merged_model", tokenizer, save_method="merged_16bit")
    
    print("\nConverting to GGUF format...")
    convert_to_gguf()

def convert_to_gguf():
    # Get absolute path of current directory
    current_dir = os.path.abspath(os.getcwd())
    base_filename = "gemma2b-q4-m.gguf"
    output_file = os.path.join(current_dir, base_filename)
    
    # Add timestamp if file exists
    if os.path.exists(output_file):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"gemma2b-q4-m_{timestamp}.gguf"
        output_file = os.path.join(current_dir, base_filename)
    
    commands = []
    
    # Check if llama.cpp directory exists
    if not os.path.exists("llama.cpp"):
        commands.append("git clone --recursive https://github.com/ggerganov/llama.cpp")
    
    # Add CMake build commands
    commands.extend([
        "cd llama.cpp && mkdir -p build && cd build && cmake .. && cmake --build . --config Release",
        "pip install gguf protobuf",
        f"python llama.cpp/convert_hf_to_gguf.py merged_model --outfile {base_filename} --outtype f16"  # Using the timestamped filename
    ])
    
    for cmd in commands:
        try:
            print(f"\nExecuting: {cmd}")
            result = subprocess.run(
                cmd, 
                shell=True, 
                check=True,
                text=True,
                capture_output=True
            )
            print(f"Output: {result.stdout}")
            print(f"Completed: {cmd}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing: {cmd}")
            print(f"Error output: {e.stderr}")
            print(f"Standard output: {e.stdout}")
            print(f"Working directory during error: {os.getcwd()}")
            break
    
    if os.path.exists(output_file):
        print(f"\nModel successfully converted and saved as: {output_file}")
        print(f"Full path: {os.path.abspath(output_file)}")
    else:
        print("\nError: Model conversion failed - output file not found")
        print(f"Expected location was: {output_file}")
        try:
            directory_contents = subprocess.run(
                f"ls -la {os.path.dirname(output_file)}", 
                shell=True, 
                text=True, 
                capture_output=True
            )
            print(directory_contents.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error listing directory: {e.stderr}")
            
    try:
        print("\nAvailable disk space:")
        df_output = subprocess.run("df -h .", shell=True, text=True, capture_output=True)
        print(df_output.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error checking disk space: {e.stderr}")

if __name__ == "__main__":
    main()