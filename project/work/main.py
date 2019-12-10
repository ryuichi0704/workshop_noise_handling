import glob

import common
from reporter.reporter import Reporter
from runner.base_runner import BaseRunner
from runner.distillation_runner import DistillationRunner
from runner.mixup_runner import MixupRunner

if __name__ == "__main__":
    if common.args.mixup:
        runner = MixupRunner()
    elif common.args.distillation:
        runner = DistillationRunner()
    else:
        runner = BaseRunner()

    runner.train_model()

    reporter = Reporter(runner)
    reporter.plot_loss_curve()

    if glob.glob("./reporter/*.json"):
        reporter.write_spreadsheet()
    else:
        print("You need to put credential file if you want to use spreadsheet api.")
