import os
import wandb

class WandbLogger():
    def __init__(self, project, is_used, name=None, entity=None):
            """
            Initializes an instance of the WandbLogger class.

            Args:
                project (str): The name of the project.
                is_used (bool): Indicates whether the instance will be used or not.
                name (str, optional): The name of the instance. Defaults to None.
                entity (str, optional): The entity associated with the instance. Defaults to None.
            """
            self.is_used = is_used
            if is_used:
                wandb.init(project=project, entity=entity, name=name)

    def watch_model(self, model):
        """
        Watches the given model using WandB if it is being used.

        Args:
            model: The model to be watched.

        Returns:
            None
        """
        if self.is_used:
            wandb.watch(model)

    def log_hyperparams(self, params):
            """
            Logs the hyperparameters to the logging service.

            Args:
                params (dict): A dictionary containing the hyperparameters to be logged.
            """
            if self.is_used:
                wandb.config.update(params)

    def log_metrics(self, metrics):
            """
            Logs the given metrics using the wandb library if the logger is being used.

            Args:
                metrics: A dictionary containing the metrics to be logged.

            Returns:
                None
            """
            if self.is_used:
                wandb.log(metrics)

    def log(self, key, value, round_idx):
        """
        Logs the given key-value pair and round index to WandB.

        Args:
            key (str): The key to log.
            value: The value to log.
            round_idx (int): The round index.

        Returns:
            None
        """
        if self.is_used:
            wandb.log({key: value, "Round": round_idx})

    def log_str(self, key, value):
        """
        Logs a key-value pair using the wandb library if self.is_used is True.

        Parameters:
            key (str): The key to log.
            value: The value to log.

        Returns:
            None
        """
        if self.is_used:
            wandb.log({key: value})


    def save_file(self, path):
        """
        Saves the file at the specified path if it exists and the object is being used.

        Args:
            path (str): The path where the file should be saved.

        Returns:
            None
        """
        if path is not None and os.path.exists(path) and self.is_used:
            wandb.save(path)