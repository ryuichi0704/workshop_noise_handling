import logging
from time import sleep

from googleapiclient import discovery, errors

logging.basicConfig(level=logging.INFO)


class ML(object):
    def __init__(self):
        self._ml = discovery.build("ml", "v1", cache_discovery=False).projects()


class Job(ML):
    def __init__(self, project):
        super().__init__()
        self._ml_jobs = self._ml.jobs()
        self._project_id = "projects/{}".format(project)


class TrainingJob(Job):
    def __init__(self, project):
        super().__init__(project)

    def execute(self, job_id, image_uri, instance_type, args):
        training_inputs = {
            "scaleTier": "CUSTOM",
            "masterType": instance_type,
            "masterConfig": {"imageUri": image_uri},
            "args": args,
            "region": "us-east1",
        }
        job_spec = {"jobId": job_id, "trainingInput": training_inputs}
        job_name = "{}/jobs/{}".format(self._project_id, job_id)

        try:
            job = self._ml_jobs.create(body=job_spec, parent=self._project_id).execute()
            logging.info(job)
            logging.info(
                "The command to trace the log is\n  gcloud ai-platform jobs stream-logs {}".format(
                    job_id
                )
            )
            logging.info(
                "The command to cancel the job is\n  gcloud ai-platform jobs cancel {}\n".format(
                    job_id
                )
            )

            return self.result(job_name)
        except errors.HttpError as err:
            raise RuntimeError(
                "There was an error creating the training job. Check the details: {}".format(
                    err._get_reason()
                )
            )
        except (TimeoutError, KeyboardInterrupt) as e:
            logging.error("this job was cancelled...")
            self._ml_jobs.cancel(name=job_name).execute()
            raise e

    def result(self, job_name):
        while True:
            result = self._ml_jobs.get(name=job_name).execute()

            if result["state"] == "SUCCEEDED":
                logging.info(result)

                return result
            elif result["state"] in ["FAILED", "CANCELLED"]:
                raise RuntimeError("job failed: {}".format(result["errorMessage"]))

            logging.info("waiting for job result... status:{}".format(result["state"]))
            sleep(10)
