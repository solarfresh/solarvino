import json
import numpy as np
from typing import Dict


def dump_beautiful_json(record: Dict, path: str):
    def convert(o):
        if isinstance(o, np.generic): return o.item()
        raise TypeError

    # now write output to a file
    json_file = open(path, "w")
    # magic happens here to make it pretty-printed
    json_file.write(json.dumps(record, indent=4, sort_keys=True, default=convert))
    json_file.close()
