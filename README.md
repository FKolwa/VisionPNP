# How to install
Just run
python -m pip install -e . --user
for python 2.7 or
python3 -m pip install -e . --user
for python >= 3.0

# How to use
import VisionPNP and access methods by VisionPNP.<METHOD_NAME>(<PARAMS>)

# Troubleshooting
## Can't find library / missing symbol import error:

Missing library has not been added to the library path or is otherwise unknown and can't be linked.
The setup.py finishes without errors but throws an import error once imported into a python script.

Solution 1 - Add library path to ldconfig:

Add path to opencv libraries to LD_LIBRARY_PATH:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib64/

Check availability:
ldconfig -N -v $(sed 's/:/ /g' <<< $LD_LIBRARY_PATH) | grep opencv

NOTE:
Has to be redone on system restart when it is not permanently added to the search paths!
Library has to be of the same version as the one used to compile the project.