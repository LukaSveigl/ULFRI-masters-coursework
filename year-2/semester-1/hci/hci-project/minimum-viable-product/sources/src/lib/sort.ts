type User = {
    displayName: string;
    username: string;
};

type Group = {
    id: string;
    displayName: string;
};

export function sortUsers(a: User, b: User) {
    const nameCompare = a.displayName.trim().localeCompare(b.displayName.trim(), "en", {
        numeric: true,
        sensitivity: "base"
    });
    if (nameCompare != 0) return nameCompare;
    if (a.username < b.username) return -1;
    if (a.username > b.username) return 1;
    return 0;
}

export function sortGroups(a: Group, b: Group) {
    const nameCompare = a.displayName.trim().localeCompare(b.displayName.trim(), "en", {
        numeric: true,
        sensitivity: "base"
    });
    if (nameCompare != 0) return nameCompare;
    if (a.id < b.id) return -1;
    if (a.id > b.id) return 1;
    return 0;
}
