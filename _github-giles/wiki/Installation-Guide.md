# Clone the public repository

The first step is to clone the public repository on your local machine in a
working directory of your choice.
In a terminal run the following commands
``` bash
git clone https://github.com/jerome-roeser/watergenics-coding-test.git
cd watergenics-coding-test
```

# Setup a virtual environment
Once you have cloned the repository and you are located in the root folder. You
should setup an isolated virtual environement and install the required dependencies.

## 1. With `pyenv` and `poetry`

If you use a tool like `pyenv` to manage different Python versions, you can switch the current python of your shell and Poetry will use it to create the new environment.

<details open>
  <summary> Check here for links to install those tools </summary>

[pyenv install](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation)

[poetry install](https://python-poetry.org/docs/#installing-with-pipx)

</details>
<br>
For instance, if your project requires a newer Python than is available with your system, a standard workflow would be:

``` bash
pyenv install 3.12.4
pyenv local 3.12.4  # Activate Python 3.12 for the current project
poetry install
```

### Activating the environment
The `poetry env activate` command prints the activate command of the virtual environment to the console. You can run the output command manually or feed it to the eval command of your shell to activate the environment. This way you won’t leave the current shell.

``` bash
eval $(poetry env activate)
```

## 2. With a `pyenv` local environment
You can also setup a separate pyenv local `pyenv` environment and use the `requirements.txt` file to install the dependencies with `pip install`
``` bash
pyenv install 3.12.4
pyenv virtualenv 3.12.4 watergenics
pyenv local watergenics  
pip install -r requirements.txt
```

## 3. With another setup
If you manage your python versions with a different tool and use a different
virtual environment tool, make sure you use a **python version >= 3.12** with the
dependencies listed in `requirements.txt` installed
