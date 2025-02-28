import os
from openai import OpenAI
import re
import asyncio
import uuid
from new1 import execute_in_jupyter

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENAI_API_KEY"),
)

chem_cont ="""Here’s a compressed version of the 10 unique RDKit use cases, retaining all essential information while being concise and suitable as context for other models:
    ---

    ### **11. 3D Visualization of Molecules**
    - **Problem**: Visualize 3D molecular shapes.
    - **Solution**: Use `py3Dmol`.
    - **Code**:
    python
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import py3Dmol
    mol = Chem.MolFromSmiles("C(C1C(C(C(C(O1)O)O)O)O)O")  # Glucose
    mol_h = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
    view = py3Dmol.view()
    view.addModel(Chem.MolToMolBlock(mol_h),'mol')
    view.setStyle({'stick':{}}).setBackgroundColor('0xeeeeee').zoomTo().show()
    

    ---

    Each example addresses a distinct problem type, leveraging RDKit’s versatility in cheminformatics, visualization, and data analysis."""

class Chemistry:
    def __init__(self):
        pass

    def extract_largest_code_block(self, text):
        code_blocks = re.findall(r'```python(.*?)```', text, re.DOTALL)
        if not code_blocks:
            return None
        return max(code_blocks, key=len)

    def generate(self, question):
        completion = client.chat.completions.create(
            model="google/gemini-2.0-flash-thinking-exp:free",
                messages=[{
                    "role":"system",
                    "content":"You are an rdkit expert that uses the rdkit library to write the necessary code in python to answer the user queries. Make sure all the answers are right and the code should be small and straight to the point.\nMore importantly, generate only required code.\n\n some examples are as follows:\n\n" + chem_cont
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        ).choices[0].message.content

        # print(completion)

        completion2 = client.chat.completions.create(
        model="google/gemini-2.0-flash-thinking-exp:free",
        messages = [{
                "role":"system",
                "content":"You are an AI assistant that analyzes and corrects the code generated and returns the detailed explanation of the output of the code but not the actual code along with corrected code. Always give me the full code at the top of the response. Also make sure all the ouputs(image/text) of the code is stored in 'chem' folder."
            },
            {
                "role":"user",
                "content": completion
            }]
        ).choices[0].message.content


        code = self.extract_largest_code_block(completion2)
        # out, err = await self.run_manim_code(class_name, code)
        if str(code).find('py3Dmol') != -1:
            code += "\nprint(Chem.MolToMolBlock(mol_h))\nprint(view.uniqueid)"
        print(code)
        out, err = execute_in_jupyter(str(code))
        if str(code).find('py3Dmol') != -1:
            a,b = out.split('END')
            a += "END"
            b = b.strip('\n').strip()
            return (a,b)
        else:
            return out
            
