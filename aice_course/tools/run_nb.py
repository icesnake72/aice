import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

path = sys.argv[1]
nb = nbformat.read(path, as_version=4)
km = KernelManager(kernel_cmd=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"])
client = NotebookClient(nb, timeout=300, km=km, resources={"metadata": {"path": "."}})
client.execute()
nbformat.write(nb, path)
errors = [(i, o) for i, c in enumerate(nb.cells) if c.cell_type == "code"
          for o in c.get("outputs", []) if o.get("output_type") == "error"]
print("executed cells:", sum(1 for c in nb.cells if c.cell_type == "code"), "| errors:", len(errors))
for i, o in errors:
  print(i, o["ename"], o["evalue"])
