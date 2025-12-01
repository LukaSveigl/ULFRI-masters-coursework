<script lang="ts">
    import AppIcon from "$lib/components/AppIcon.svelte";
    import Avatar from "$lib/components/Avatar.svelte";
    import Popup from "$lib/components/Popup.svelte";
    import { onGetMembers, onLeaveGroup } from "$lib/components/PopupDetailsGroup.telefunc";
    import PopupLeaveGroup from "$lib/components/PopupLeaveGroup.svelte";
    import { Button } from "$lib/shadcn/components/ui/button";
    import { ScrollArea } from "$lib/shadcn/components/ui/scroll-area";
    import { sortUsers } from "$lib/sort";
    import { HandCoins } from "lucide-svelte";

    let {
        open = $bindable(false),
        group,
        onleavegroup
    }: {
        open?: boolean;
        group: Readonly<{
            id: string;
            displayName: string;
            avatar: boolean;
            avatarModified: Date;
        }>;
        onleavegroup?: () => Promise<void>;
    } = $props();

    let openRemove = $state(false);
    $effect.pre(() => {
        if (!open) openRemove = false;
    });
</script>

<Popup bind:open>
    <div class="flex items-center gap-4">
        <Avatar user={group} isGroup />
        <p>{group.displayName}</p>
    </div>
    <div>
        <p class="text-xl">Members</p>
        <ScrollArea orientation="vertical" class="h-40">
            {#await onGetMembers(group.id) then users}
                {@const sortedUsers = users.toSorted(sortUsers)}
                <ul class="flex flex-col">
                    {#each sortedUsers as user (user.id)}
                        <li
                            class="flex items-center gap-4 border-t border-inherit py-2 last:border-b"
                        >
                            <Avatar {user} />
                            <p>{user.displayName}</p>
                        </li>
                    {/each}
                </ul>
            {/await}
        </ScrollArea>
    </div>
    <hr />
    <Button class="shadow-md" href={"/issueloan?" + new URLSearchParams({ group: group.id })}>
        <AppIcon icon={HandCoins} class="!size-6" />
        Issue Loan
    </Button>
    <PopupLeaveGroup
        bind:open={openRemove}
        {group}
        onconfirm={async () => {
            await onLeaveGroup(group.id);
            await onleavegroup?.();
            openRemove = false;
        }}
    />
</Popup>
