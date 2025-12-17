
# .

Top level files

##### ./README.md

Top level README.md file.  Includes

* Badges
  * Tests
  * Documentation
  * pypi
  * License
  * Python versions supported 
* Overview section
  * Key Features
* Installation
* Documentation


##### ./CONTRIBUTING.md

Description of how to contribute / dev guide.

* Code of conduct
* Getting started
* Development workflow
  * Creating a branch
  * Making changes
* Code style
  * Tools
  * Commit messages
* Documentation
* Submitting changes
* Review process
* Types of contributions
* Release process
* Questions
* License


##### ./RELEASING.md

* Release instructions
  * Prerequisites
  * Release Process
  * Testing on Test PyPI


##### ./LICENSE

MIT License


##### ./.gitignore

Standard python .gitignore with a few add ons


##### ./pyproject.toml

Project file with the following sections

* build-system
* project
* tool.setuptools
* tool.black
* tool.mypy
* tool.pylint
* tool.coverage


## src/c2i2o

Source files

##### src/c2i2o/__init__.py

Top level init file.  Includes version and anything lifted to top level.


### src/c2i2o/core

Core library files.  Includes all type defintions and base classes.


##### src/c2i2o/core/__init__.py

Core library init file.  Lift all base classes to here.


##### src/c2i2o/core/core_utils.py



##### src/c2i2o/core/distributions.py

Base of distributions classes.


###### src/c2i2o/core/distributions::DistributionBase

Inherits from pydantic.BaseModel

Implements sample() and log_pdf()

##### src/c2i2o/core/scipy_distributions.py

Implements several distributions that all inhert from DistributionBase

###### src/c2i2o/core/scipy_distributions::Norm
###### src/c2i2o/core/scipy_distributions::Lognorm
###### src/c2i2o/core/scipy_distributions::Expon


##### src/c2i2o/core/parameter.py

###### src/c2i2o/core/parameter:Parameter


##### src/c2i2o/core/parameterSpace.py

###### src/c2i2o/core/parameterSpace::ParameterSpace


##### src/c2i2o/core/grid.py

###### src/c2i2o/core/grid::Grid


##### src/c2i2o/core/intermediate.py

###### src/c2i2o/core/intermediate::IntermediateBase

###### src/c2i2o/core/intermediate::MatterPowerSpectrum

###### src/c2i2o/core/intermediate::ComovingDistance

###### src/c2i2o/core/intermediate::HubbleEvolution


##### src/c2i2o/core/tracer.py

##### src/c2i2o/core/tracer:Tracer


##### src/c2i2o/core/observable.py

###### src/c2i2o/core/observable::ObservableBase

###### src/c2i2o/core/observable::DistanceMeasure

###### src/c2i2o/core/observable::CrossCorrlation

###### src/c2i2o/core/observable::AngularCrossCorrlation

###### src/c2i2o/core/observable::HarmonicCrossCorrlation


##### src/c2i2o/core/data_vector.py

###### src/c2i2o/core/data_vector:DataVector


##### src/c2i2o/core/emulation.py

###### src/c2i2o/core/emulation:EmulatorBase


##### src/c2i2o/core/encoding.py

###### src/c2i2o/core/encoding:EncoderBase


##### src/c2i2o/core/inference.py

###### src/c2i2o/core/inference:InferenceBase



### src/c2i2o/utils

##### src/c2i2o/utils/metric.py

##### src/c2i2o/utils/validation.py





