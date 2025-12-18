
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


##### src/c2i2o/core/types.py

Type defintions and enums


##### src/c2i2o/core/core_utils.py

Utilities that are used in the core package.  E.g., conversions, validation utilities.


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

###### src/c2i2o/core/intermediate::IntermediateSet


##### src/c2i2o/core/tracer.py

##### src/c2i2o/core/tracer:Tracer


##### src/c2i2o/core/observable.py

###### src/c2i2o/core/observable::ObservableBase

###### src/c2i2o/core/observable::ObservableSet

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

###### src/c2i2o/core/inference:I2CInferenceBase

###### src/c2i2o/core/inference:O2CInferenceBase

###### src/c2i2o/core/inference:O2IInferenceBase


### src/c2i2o/utils

##### src/c2i2o/utils/metric.py



### src/c2i2o/intermediates

###### src/c2i2o/intermediates/matter_power_spectrum::MatterPowerSpectrum

###### src/c2i2o/intermediates/comoving_distance::ComovingDistance

###### src/c2i2o/intermediates/hubble_evolution::HubbleEvolution



### src/c2i2o/observable





### src/c2i2o/emulation

Emulation base files

##### src/c2i2o/emulation/__init__.py

Emulation library init file.  Lift all base classes to here.

###### src/c2i2o/emulation/c2i_emulation:C2IEmulatorBase

###### src/c2i2o/emulation/i20_emulation:I2OEmulatorBase

###### src/c2i2o/emulation/c2o_emulation:C2OEmulatorBase



### src/c2i2o/encoding

Emulation base files

##### src/c2i2o/encoding/__init__.py

Emulation library init file.  Lift all base classes to here.

###### src/c2i2o/encoding/encoding:IntermediateEncoderBase

###### src/c2i2o/encoding/encoding::MatterPowerSpectrumEncoderBase

###### src/c2i2o/encoding/encoding::ComovingDistanceEncoderBase

###### src/c2i2o/encoding/encoding::HubbleEvolutionEncoderBase

###### src/c2i2o/encoding/encoding:ObservableEncodingBase



### src/c2i2o/inference

Inference base files

##### src/c2i2o/inference/__init__.py

Inference library init file.  Lift all base classes to here.

###### src/c2i2o/inference/inference:I2CInferenceBase

###### src/c2i2o/inference/inference:O2CInferenceBase

###### src/c2i2o/inference/inference:O2IInferenceBase



### src/c2i2o/interfaces

### src/c2i2o/interfaces




### src/c2i2o/db

### src/c2i2o/db/tables

##### src/c2i2o/db/tables/__init__.py

##### src/c2i2o/db/tables/base.py

##### src/c2i2o/db/tables/cosmology.py

##### src/c2i2o/db/tables/intermediates.py

##### src/c2i2o/db/tables/observables.py

##### src/c2i2o/db/tables/trained_models.py

##### src/c2i2o/db/tables/encodings.py

##### src/c2i2o/db/tables/c2i_association.py

##### src/c2i2o/db/tables/c2o_association.py

##### src/c2i2o/db/i2o_association.py

##### src/c2i2o/db/tables/c2i_models.py

##### src/c2i2o/db/tables/c2o_models.py



### src/c2i2o/db/pydantic_models


### src/c2i2o/web/db_server


