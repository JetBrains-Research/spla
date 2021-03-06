/**********************************************************************************/
/* This file is part of spla project                                              */
/* https://github.com/JetBrains-Research/spla                                     */
/**********************************************************************************/
/* MIT License                                                                    */
/*                                                                                */
/* Copyright (c) 2021 JetBrains-Research                                          */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files (the "Software"), to deal  */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions:                       */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/**********************************************************************************/

#ifndef SPLA_SPLATYPE_HPP
#define SPLA_SPLATYPE_HPP

#include <spla-cpp/SplaObject.hpp>
#include <utility>

namespace spla {

    /**
     * @addtogroup API
     * @{
     */

    /**
     * @class Type
     *
     * Represents predefined of user-defined type description.
     *
     * In the library values types is effectively a unique type string name,
     * type annotations, size in bytes and etc. Value itself is bytes memory
     * region, interpreted in the way defined by the user.
     */
    class SPLA_API Type final : public Object {
    public:
        ~Type() override = default;

        /** @return Type string name */
        const std::string &GetId() const {
            return mId;
        }

        /** @return Size in bytes of the single value of this type */
        std::size_t GetByteSize() const {
            return mByteSize;
        }

        /** @return True if it is built-in predefined type */
        bool IsBuiltIn() const {
            return mBuiltIn;
        }

        /** @return True if type has actual values, otherwise it is type without values */
        bool HasValues() const {
            return GetByteSize() != 0;
        }

        /**
         * Makes new user-defined type.
         *
         * @note Raise error if type with specified id already registered in the library.
         *
         * @param id Unique name of the type.
         * @param typeSize Size of the type values in bytes.
         * @param library Library global instance.
         *
         * @return New type instance.
         */
        static RefPtr<Type> Make(std::string id, std::size_t typeSize, class Library &library);

        /**
         * Finds already register type in library by specified id.
         *
         * @note Returns null if type with specified id is not found.
         *
         * @param id Unique name of the type.
         * @param library Library global instance.
         *
         * @return Type reference.
         */
        static RefPtr<Type> Find(std::string id, class Library &library);

    private:
        friend class Types;
        static RefPtr<Type> Make(std::string id, std::size_t typeSize, bool builtIn, class Library &library);

        Type(std::string id, std::size_t typeSize, bool builtIn, class Library &library);

        // Unique type name (id)
        std::string mId;

        // Size of the type value in bytes
        // Note: if size is 0 => Object with this type have no values
        std::size_t mByteSize = 0;

        // True for built-in (predefined) types
        bool mBuiltIn = false;
    };

    /** Inherit from this class if your object must have type info */
    class SPLA_API TypedObject : public Object {
    public:
        ~TypedObject() override = default;

        /** @return True if this and other typed object has compatible types */
        [[nodiscard]] bool IsCompatible(const TypedObject &other) const {
            return mType == other.mType;
        }

        /** @return Type info of this object */
        [[nodiscard]] const RefPtr<Type> &GetType() const {
            return mType;
        }

    protected:
        explicit TypedObject(RefPtr<Type> type, TypeName typeName, class Library &library);

    private:
        RefPtr<Type> mType;
    };

    /**
     * @}
     */

}// namespace spla

#endif//SPLA_SPLATYPE_HPP
