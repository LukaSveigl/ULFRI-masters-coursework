import { getFriends } from "$lib/server/db/common_queries";
import { encodeUser } from "$lib/server/db_utils";
import { getUser } from "$lib/server/telefunc_util";

export async function onRequestFriends() {
    const user = getUser();
    return (await getFriends(user.id)).map(encodeUser);
}
