# Repository Structure

24.10.2023

## Context

A ML model is being developed in this repository.
The repository is using the MLCore SDK, which provides infrastructure, deployment, and serving logic.

## Decision

To ensure compatibility with the MLCore SDK and abstract the infrastructure and deployment logic, the following repository structure is proposed:

- *preprocess/*: This directory will contain code and configuration files for preprocessing data before training or serving. It will be used to generate features.

13.11.2023: The containarization should be manually performed, and the container should be uploaded to a container registry. The resulting uri should be provided to the MLCore SDK when creating a model.

1.09.2025: MLCore SDK has been deprecated. The containarization is now performed using GitHub Actions, unified between steps (includes preprocess+inference+postprocess), and the container is uploaded to a container registry automatically upon pushing to the main branch. The resulting uri is provided to a MongoDB database when creating a model, and used by verdeliss.

- *postprocess/*: This directory will contain code and configuration files for post-processing and format inference outputs, both for training and serving.

13.11.2023: The containarization should be manually performed, and the container should be uploaded to a container registry. The resulting uri should be provided to the MLCore SDK when creating a model.

1.09.2025: MLCore SDK has been deprecated. The containarization is now performed using GitHub Actions, unified between steps (includes preprocess+inference+postprocess), and the container is uploaded to a container registry automatically upon pushing to the main branch. The resulting uri is provided to a MongoDB database when creating a model, and used by verdeliss.

- *inference/*: This directory will contain code and configuration files for performing inference.

13.11.2023: The containarization should be manually performed, and the container should be uploaded to a container registry. The resulting uri should be provided to the MLCore SDK when∫ creating a model.

1.09.2025: MLCore SDK has been deprecated. The containarization is now performed using GitHub Actions, unified between steps (includes preprocess+inference+postprocess), and the container is uploaded to a container registry automatically upon pushing to the main branch. The resulting uri is provided to a MongoDB database when creating a model, and used by verdeliss.

## Consequences

The *preprocess/* and *postprocess/* directories will facilitate the generation of features and the formatting of inference outputs, which are necessary for both training and serving, avoiding redundancies.
The *inference/* directory will enable the model to be deployed as a Docker container and perform inference abstracting versioning.
