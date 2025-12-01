import { Abort, getContext } from "telefunc";

export function getUser() {
    const { user } = getContext();
    if (!user) {
        throw Abort({ notLoggedIn: true });
    }
    return user;
}
