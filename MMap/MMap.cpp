// MMap.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "mmap.h"
#include <iostream>
#include <memory>

struct test {
	int i[1];
};

int main()
{

	std::shared_ptr<MMap<test>>mmap_int1 = std::make_shared<MMap<test>>("TEST_INT313", 0);

	std::cout << 1 << ":" << mmap_int1->get_error() << std::endl;
	
	MMap<test> mmap_int2("TEST_INT313", true);
	std::cout << 2 << ":" << mmap_int2.get_error() << std::endl;

	test t;
	t.i[0] = 100;
	mmap_int1->set(t);
	std::cout << 3 << ":" << mmap_int1->get_error() << std::endl;

	const volatile test* tt = mmap_int1->get_pointer();
	std::cout << 4 << ":" << tt->i[0] << std::endl;

	t.i[0] = 0;
	mmap_int2.get(t);
	std::cout << 5 << ":" << mmap_int2.get_error() << std::endl;
	std::cout << 6 << ":" << t.i[0] << std::endl;
	t.i[0] = 0;
	mmap_int2.set(t);

	std::cout << 7 << ":" << tt->i[0] << std::endl;

	MMap<char> mmap_int3("TEST_INT313", 0);
	std::cout << 8 << ":" << mmap_int3.get_error() << std::endl;

	int r;
	std::cin>> r;
    return 0;
}

