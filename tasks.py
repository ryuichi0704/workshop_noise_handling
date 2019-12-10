import os
from datetime import datetime
from uuid import uuid4

from invoke import task

from ai_platform import TrainingJob


# ------Local Commands------
@task
def build_local(c):
    c.run(
        "docker build . -f image/local/Dockerfile -t {tag} --build-arg USER_NAME=$USER".format(
            tag=c.config.local.tag
        )
    )


@task
def stop_local(c):
    c.run("docker stop train_container")


@task
def run_local(c):
    cmd = "docker run --runtime='nvidia' -itd --name train_container \
        -v $PWD/project:/home/$USER -v /etc/group:/etc/group:ro \
        -v /etc/passwd:/etc/passwd:ro --ipc=host \
        -u $(id -u $USER):$(id -g $USER) --rm {tag}".format(
        tag=c.config.local.tag
    )

    print(cmd)
    c.run(cmd)


@task
def attach_local(c):
    c.run("docker exec -it train_container /bin/bash", pty=True)


# ------Remote Commands------
def set_env(c):
    os.environ["GOOGLE_CLOUD_PROJECT"] = c.config.remote.gcp_project
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = c.config.remote.credential_path


@task
def push_remote(c):
    set_env(c)
    c.run(
        "docker build . -f image/remote/Dockerfile -t {tag}".format(
            tag=c.config.remote.tag
        )
    )

    remote_tag = "gcr.io/{repo}/{tag}".format(
        repo=c.config.remote.gcp_project, tag=c.config.remote.tag
    )
    c.run(
        "docker tag {tag} {remote_tag}".format(
            tag=c.config.remote.tag, remote_tag=remote_tag
        )
    )
    c.run("docker push {}".format(remote_tag))


@task
def train_ai_platform(
    c,
    batch_size=None,
    learning_rate=None,
    epoch=None,
    mixup=False,
    manifold_mixup=False,
    mixup_alpha=0.2,
    momentum=None,
    weight_decay=None,
    distillation=False,
    soft_loss_weight=None,
    softmax_temperature=None,
    seed=None,
    uuid=None,
):
    set_env(c)
    training_job = TrainingJob(c.config.remote.gcp_project)
    instance_type = c.config.remote.instance_type
    image_uri = "gcr.io/{repo}/{tag}".format(
        repo=c.config.remote.gcp_project, tag=c.config.remote.tag
    )
    job_id = "train_{datetime}_{str_uuid}".format(
        datetime=datetime.now().strftime("%Y%m%d%H%M%S"),
        str_uuid=str(uuid4()).replace("-", "_"),
    )

    arguments = ["--gpu", "0"]

    if batch_size:
        arguments.append("--batch-size")
        arguments.append(str(batch_size))

    if learning_rate:
        arguments.append("--lr")
        arguments.append(str(learning_rate))

    if epoch:
        arguments.append("--epoch")
        arguments.append(str(epoch))

    if mixup:
        arguments.append("--mixup")
        arguments.append("--mixup-alpha")
        arguments.append(str(mixup_alpha))

    if manifold_mixup:
        arguments.append("--manifold-mixup")
        arguments.append("--mixup-alpha")
        arguments.append(str(mixup_alpha))

    if momentum:
        arguments.append("--momentum")
        arguments.append(str(momentum))

    if weight_decay:
        arguments.append("--weight-decay")
        arguments.append(str(weight_decay))

    if distillation:
        arguments.append("--distillation")
        if soft_loss_weight:
            arguments.append("--soft-loss-weight")
            arguments.append(str(soft_loss_weight))
        if softmax_temperature:
            arguments.append("--softmax-temperature")
            arguments.append(str(softmax_temperature))

    if seed:
        arguments.append("--seed")
        arguments.append(str(seed))

    if uuid:
        arguments.append("--uuid")
        arguments.append(str(uuid))

    training_job.execute(job_id, image_uri, instance_type, arguments)
