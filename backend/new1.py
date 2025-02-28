import time
from jupyter_client import KernelManager

def execute_in_jupyter(code: str) -> tuple:
    """
    Executes Python code inside a Jupyter kernel and returns stdout and stderr.
    
    Args:
        code (str): The Python code to execute.
    
    Returns:
        tuple: (stdout, stderr) captured from execution.
    """
    km = KernelManager(kernel_name='python3')  # <-- Key fix here
    km.start_kernel()
    kc = km.client()
    kc.start_channels()

    # Allow time for kernel initialization
    time.sleep(2)

    stdout, stderr = [], []

    # Install required packages with proper imports and wait
    install_code = """import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'rdkit', 'py3Dmol'])
"""
    install_msg_id = kc.execute(install_code)
    install_done = False
    while not install_done:
        try:
            msg = kc.get_iopub_msg(timeout=10)
            if msg['parent_header'].get('msg_id') == install_msg_id:
                msg_type = msg.get('msg_type', '')
                content = msg.get('content', {})
                if msg_type == 'status' and content.get('execution_state') == 'idle':
                    install_done = True
                elif msg_type == 'error':
                    stderr.append('\n'.join(content.get('traceback', ['Package installation failed'])))
                    break
        except Exception as e:
            stderr.append(f"Installation error: {str(e)}")
            break

    # Execute user code only if installation succeeded
    if install_done:
        msg_id = kc.execute(code)
        while True:
            try:
                print(msg)
                msg = kc.get_iopub_msg(timeout=10)
                if msg['parent_header'].get('msg_id') == msg_id:
                    msg_type = msg.get('msg_type', '')
                    content = msg.get('content', {})
                    if msg_type == 'stream':
                        if content.get('name') == 'stdout':
                            stdout.append(content.get('text', ''))
                        elif content.get('name') == 'stderr':
                            stderr.append(content.get('text', ''))
                    elif msg_type == 'error':
                        stderr.append("\n".join(content.get('traceback', [])))
                    elif msg_type == 'status' and content.get('execution_state') == 'idle':
                        break
            except Exception as e:
                stderr.append(f"Execution error: {str(e)}")
                break
    


    # Cleanup
    try:
        kc.stop_channels()
        km.shutdown_kernel()
    except Exception as e:
        stderr.append(f"Cleanup error: {str(e)}")

    return ''.join(stdout), ''.join(stderr)