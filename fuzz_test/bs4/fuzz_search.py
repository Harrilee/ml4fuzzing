import random


def random_find_all_args():
    """Generate a set of arguments for soup.find_all() that almost always returns results."""

    # Since we know the generated HTML always has multiple <p>, <section>, and <article> tags,
    # we'll restrict to these tags to ensure a high success rate.
    possible_tags = ["p", "section", "article"]

    name = random.choice(possible_tags)

    # We know that each element has id="id-x" and class="class-x" for some x in [0,5].
    # We'll only pick attributes from these to ensure we have a high chance of matches.
    possible_attrs = ["class", "id"]

    attrs = {}
    # Let's randomly pick up to 1 attribute to narrow down (fewer attributes = higher chance of matches)
    if random.random() < 0.7:  # 70% chance to pick an attribute, can adjust as needed
        attr_name = random.choice(possible_attrs)
        # Generate id-x or class-x
        if attr_name == "id":
            attr_value = f"id-{random.randint(0, 3)}"
        else:  # class
            attr_value = f"class-{random.randint(0, 3)}"
        attrs[attr_name] = attr_value

    # We'll keep recursive=True for a thorough search
    recursive = True

    # Limit can be random, but having no limit increases chances of finding something
    limit = None

    # No text searching to avoid misses
    # No 'string' argument is given, so we rely on structure only.

    # Sometimes add keyword arguments (class_ or id_) to further ensure matches
    # Since we know the structure always has these attributes, let's do it occasionally.
    kwargs = {}
    if random.random() < 0.5:  # 50% chance to add a keyword arg
        kw_attr_name = random.choice(possible_attrs)
        if kw_attr_name == "class":
            kwargs["class_"] = f"class-{random.randint(0, 5)}"
        else:
            kwargs["id"] = f"id-{random.randint(0, 5)}"

    args = {
        "name": name,
        "attrs": attrs if attrs else None,
        "recursive": recursive,
        "limit": limit,
    }

    clean_args = {k: v for k, v in args.items() if v is not None}
    clean_args.update(kwargs)

    return clean_args


if __name__ == "__main__":
    for _ in range(10):
        print(random_find_all_args())