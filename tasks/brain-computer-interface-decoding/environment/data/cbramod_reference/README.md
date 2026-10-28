# CBraMod data-preparation reference

These two unchanged preprocessing scripts come from
[CBraMod commit b9e961003214326972c567eff390e75b0287e32a](https://github.com/wjq-learning/CBraMod/tree/b9e961003214326972c567eff390e75b0287e32a/preprocessing).
The accompanying [MIT license](LICENSE) is retained. They document the FACED
and TUEV source-to-processed recipes, including their original hard-coded
paths; configure paths before running them on authorized source data.

The task's [materializer](../materialize.py) consumes the user's already
processed FACED LMDB and final TUEV pickles. Its TUEV train/validation
subject regrouping is a separate task-side step; neither original script
contains this adaptation. The classifier baseline uses Braindecode 1.8.1,
including its `CBraMod` model, and does not import this reference code.
