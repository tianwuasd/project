import ast
import difflib
import importlib.util
import json
from pathlib import Path
from urllib.request import ProxyHandler, build_opener


SYMBOLS = {
    "networks/nets/senet.py": ["SENet", "SENet154"],
    "networks/blocks/squeeze_and_excitation.py": ["ChannelSELayer", "SEBlock", "SEBottleneck"],
    "networks/blocks/convolutions.py": ["Convolution"],
    "networks/blocks/acti_norm.py": ["ADN"],
    "networks/layers/factories.py": ["LayerFactory", "split_args", "conv_factory", "batch_factory",
                                     "dropout_factory", "maxpooling_factory", "adaptive_avgpooling_factory"],
    "networks/layers/utils.py": ["get_act_layer", "get_norm_layer", "get_dropout_layer"],
    "networks/layers/convutils.py": ["same_padding", "stride_minus_kernel_padding"],
}


class ComputationOnly(ast.NodeTransformer):
    def visit_arg(self, node):
        node.annotation = None
        return node

    def visit_FunctionDef(self, node):
        node.returns = None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value is None:
            return ast.Pass()
        return self.generic_visit(ast.Assign(targets=[node.target], value=node.value))

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)


def definitions(source):
    tree = ComputationOnly().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    return {item.name: item for item in tree.body if isinstance(item, (ast.ClassDef, ast.FunctionDef))}


def main():
    root = Path(importlib.util.find_spec("monai").origin).parent
    opener = build_opener(ProxyHandler({"https": "http://127.0.0.1:7897", "http": "http://127.0.0.1:7897"}))
    result = []
    for relative, names in SYMBOLS.items():
        url = "https://raw.githubusercontent.com/Project-MONAI/MONAI/1.4.0/monai/" + relative
        with opener.open(url, timeout=30) as response:
            original = definitions(response.read().decode("utf-8"))
        current = definitions((root / relative).read_text(encoding="utf-8"))
        for name in names:
            left, right = ast.dump(original[name], include_attributes=False), ast.dump(current[name], include_attributes=False)
            item = {"path": relative, "symbol": name, "computation_ast_equal": left == right, "source": url}
            if left != right:
                item["difference"] = list(difflib.unified_diff(ast.unparse(original[name]).splitlines(),
                                                              ast.unparse(current[name]).splitlines(), n=2))
            result.append(item)
    print(json.dumps({"scope": "source_only_no_model_instance_or_forward", "local_root": str(root),
                      "ignored": ["docstrings", "type_annotations", "source_locations"], "symbols": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
