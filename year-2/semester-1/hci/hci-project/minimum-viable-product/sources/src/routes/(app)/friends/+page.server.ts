import { getFriends } from "$lib/server/db/common_queries";
import { encodeUser } from "$lib/server/db_utils";
import type { PageServerLoad } from "./$types";

export const load: PageServerLoad = async (event) => {
    await event.parent();
    const user = event.locals.user!;
    return { friends: (await getFriends(user.id)).map(encodeUser) };
};
