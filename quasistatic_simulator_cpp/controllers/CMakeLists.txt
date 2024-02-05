add_library(contact_model contact_model.h contact_model.cc)
target_link_libraries(contact_model drake::drake)
target_include_directories(contact_model PUBLIC ${PROJECT_SOURCE_DIR})

add_library(contact_controller
        contact_controller.h
        contact_controller.cc
        contact_model.h
        contact_model.cc)
target_link_directories(contact_controller PUBLIC)
target_link_libraries(contact_controller contact_model drake::drake)
target_include_directories(contact_controller PUBLIC 
        ${PROJECT_SOURCE_DIR})
