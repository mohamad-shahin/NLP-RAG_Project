Put your trained model file here as:

    model/retinal_model.h5

The app looks for exactly this filename. See the main project README for
how to produce this file from the training notebook, and for an alternative
(recommended) approach of hosting the file externally and setting the
`MODEL_URL` environment variable instead of committing a large binary to git.
