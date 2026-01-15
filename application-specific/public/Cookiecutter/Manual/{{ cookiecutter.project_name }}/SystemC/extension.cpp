#include "Target.h"

struct my_extension : tlm_extension<my_extension>
{
    // Attributes
    int id;
    // constructor and default values
    my_extension() : id(0) {}

    // must write this 2 function for creating an extension "custom payload"
    tlm_extension_base *clone() const
    {
        my_extension *t = new my_extension;
        t->id = this->id;
        return t;
    }
    virtual void copy_from(tlm_extension_base const &ext)
    {
        id = static_cast<my_extension const &>(ext).id;
    }
};