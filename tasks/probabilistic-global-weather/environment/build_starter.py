"""Build an explicit agent-facing runtime; adaptation diagnostics stay outside."""
import argparse
import ast
from pathlib import Path
import shutil

STARTER = ('native_model.py','training_data.py','train_stage.py','plan_schedule.py',
           'ar_extension.py','predict.py','inference_state.py','forecast_store.py','package_model.py','training-protocol.json',
           'protocol.json','precipitation-protocol.json','LICENSE.weathernext-code')
EVALUATION = ('development.py','evaluate.py','merge_scores.py','training_data.py',
              'namespace_inference.py','zarr_safety.py','development-protocol.json')


class RuntimeOnly(ast.NodeTransformer):
    def __init__(self, name):self.name=name
    def visit_Module(self,node):
        remove={'wn2_validation.py':{'main','eligible'},'wn2_distributed.py':{'make_step','run_probe'},
                'plan_schedule.py':{'plan','main'}}.get(self.name,set())
        node.body=[n for n in node.body if not (isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in remove)
                   and not (remove and isinstance(n,ast.If) and '__name__' in ast.unparse(n.test))]
        return self.generic_visit(node)
    def visit_Expr(self,node):
        c=node.value
        if isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and c.func.attr=='add_argument' and c.args:
            flag=c.args[0].value if isinstance(c.args[0],ast.Constant) else None
            if self.name=='train_stage.py' and flag in ('--profile','--debug-single-device','--debug-updates'):return None
            if self.name=='package_model.py' and flag=='--allow-profile':return None
            if self.name=='train_stage.py' and flag=='--schedule':c.keywords.append(ast.keyword(arg='required',value=ast.Constant(True)))
        return self.generic_visit(node)
    def visit_Assign(self,node):
        c=node.value
        if isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and c.func.attr=='parse_args':
            defaults={'train_stage.py':'parser.set_defaults(profile=False, debug_single_device=False, debug_updates=1)',
                      'package_model.py':'parser.set_defaults(allow_profile=False)'}.get(self.name)
            if defaults:return [ast.parse(defaults).body[0],node]
        return self.generic_visit(node)

def build(root, output):
    code=root/'environment'/'weathernext'  # shared WeatherNext 2 sources: wn2/, slurm/, paper-comparison/
    starter=output/'starter';evaluation=output/'evaluation'
    starter.mkdir(parents=True);evaluation.mkdir(parents=True)
    for name in STARTER:shutil.copyfile(code/'wn2'/name,starter/name)
    for name in EVALUATION:shutil.copyfile(code/'wn2'/name,evaluation/name)
    for name in ('wn2_validation.py','wn2_distributed.py'):shutil.copyfile(code/'slurm'/name,starter/name)
    for name in ('wn2_validation.py','wn2_distributed.py','plan_schedule.py','train_stage.py','package_model.py'):
        p=starter/name;tree=RuntimeOnly(name).visit(ast.parse(p.read_text()));p.write_text(ast.unparse(ast.fix_missing_locations(tree))+'\n')
    for p in (root/'environment/baseline/starter').iterdir():
        if p.is_file():shutil.copyfile(p,starter/p.name)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.root,args.output)
