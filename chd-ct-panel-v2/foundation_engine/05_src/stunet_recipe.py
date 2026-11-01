"""One scientific contract for new STU-Net-S jobs; no runtime side effects.

Official released nnUNet-2.2 small/small_ft trainers, pinned at 0652eaf5.
Global batch is an explicit user choice (default 4), not a paper batch claim.
Historical frozen 250-epoch jobs execute their own unchanged source.
"""
RECIPE_ID='stunet_s_official_code_1000_v1'
UPSTREAM_COMMIT='0652eaf5cafda6bc269942238c3695994f2dc4f0'
EPOCHS=1000
TRAIN_ITERATIONS=250
VALIDATION_ITERATIONS=50


def budget():
    return dict(epochs=EPOCHS,train_iterations=TRAIN_ITERATIONS,validation_iterations=VALIDATION_ITERATIONS)


def recipe(method):
    if method not in {'S1','S0'}:raise ValueError('STU-Net配方只支持S1/S0')
    return dict(id=RECIPE_ID,upstream_commit=UPSTREAM_COMMIT,method=method,
        **budget(),initial_lr=1e-3 if method=='S1' else 1e-2,
        optimizer='SGD',momentum=.99,nesterov=True,weight_decay=3e-5,
        scheduler='PolyLRScheduler',poly_exponent=.9,
        primary_checkpoint='checkpoint_final.pth',compatibility_port='nnunetv2-2.8.1')


def require_trainer_budget(trainer):
    actual=(trainer.num_epochs,trainer.num_iterations_per_epoch,trainer.num_val_iterations_per_epoch)
    if actual!=(EPOCHS,TRAIN_ITERATIONS,VALIDATION_ITERATIONS):
        raise ValueError('正式STU-Net预算与官方1000轮×250训练/50验证不符')


def require_completed_recipe(run,method):
    if (run.get('status')!='completed' or not run.get('formal_training')
            or run.get('signature',{}).get('training_recipe')!=recipe(method)
            or run.get('planned_epochs')!=EPOCHS or run.get('completed_epochs')!=EPOCHS):
        raise ValueError('来源不是当前官方1000轮配方的完整模型；旧250轮任务保持冻结版本，请选择兼容训练任务')
