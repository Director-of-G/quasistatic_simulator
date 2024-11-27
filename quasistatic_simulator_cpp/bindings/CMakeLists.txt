pybind11_add_module(qsim_cpp MODULE qsim_cpp.cc)

# target_link_libraries(qsim_cpp PUBLIC quasistatic_simulator pinocchio_calculator ${PINOCCHIO_LIBRARIES} /opt/openrobots/lib/libpinocchio_parsers.so)
target_link_libraries(qsim_cpp PUBLIC quasistatic_simulator pinocchio::pinocchio)

set_target_properties(qsim_cpp PROPERTIES CXX_VISIBILITY_PRESET default)

pybind11_add_module(contact_ctrl_cpp MODULE contact_ctrl_cpp.cc)
target_link_libraries(contact_ctrl_cpp PUBLIC contact_controller)
set_target_properties(contact_ctrl_cpp PROPERTIES CXX_VISIBILITY_PRESET default)

