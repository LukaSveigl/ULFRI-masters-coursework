<script lang="ts">
    import { invalidateAll } from "$app/navigation";
    import AppIcon from "$lib/components/AppIcon.svelte";
    import Avatar from "$lib/components/Avatar.svelte";
    import Label from "$lib/components/Label.svelte";
    import Popup from "$lib/components/Popup.svelte";
    import PopupDetailsFriend from "$lib/components/PopupDetailsFriend.svelte";
    import { Button, buttonVariants } from "$lib/shadcn/components/ui/button";
    import * as Dialog from "$lib/shadcn/components/ui/dialog";
    import { Input } from "$lib/shadcn/components/ui/input";
    import { ScrollArea } from "$lib/shadcn/components/ui/scroll-area";
    import { sortUsers } from "$lib/sort.js";
    import { mutation } from "$lib/utils.svelte.js";
    import { Plus } from "lucide-svelte";
    import { onAddFriend } from "./page.telefunc.js";

    let { data } = $props();

    let filter = $state("");
    let filterLower = $derived(filter.toLowerCase());
    let filteredFriends = $derived(
        data.friends
            .filter((x) => {
                if (filter == "") return true;
                if (filter == filterLower) {
                    return x.displayName.toLowerCase().includes(filterLower);
                }
                return x.displayName.includes(filter);
            })
            .toSorted(sortUsers)
    );

    let detailsDialog: { open: boolean; user: (typeof data.friends)[number] } = $state({
        open: false,
        user: undefined!
    });

    let addFriendOpen = $state(false);
    let addFriendUsername = $state("");
    let addFriendError: string | null = $state(null);

    const mutAddFriend = mutation(async (username: string) => {
        const { ok, message } = await onAddFriend(username);
        if (ok) {
            addFriendError = null;
            await invalidateAll();
            addFriendOpen = false;
        } else {
            addFriendError = message;
        }
    });

    let addFriendFullErr = $derived(
        mutAddFriend.stalled == "error" ? String(mutAddFriend.stallError) : addFriendError
    );
</script>

<div class="relative h-full">
    <ScrollArea
        orientation="vertical"
        class="flex h-full flex-col items-stretch rounded-md border-2 border-neutral-300"
    >
        <div class="m-4">
            <Input type="search" placeholder="Search" bind:value={filter} />
        </div>
        <ul class="mx-4 mt-2 flex flex-col">
            {#each filteredFriends as friend (friend.id)}
                <li class="group contents">
                    <button
                        class="flex items-center gap-4 border-t border-inherit py-2 group-last:border-b"
                        onclick={() => (detailsDialog = { open: true, user: friend })}
                    >
                        <Avatar user={friend} />
                        <p>{friend.displayName}</p>
                    </button>
                </li>
            {/each}
        </ul>
        <div class="h-16 my-2"></div>
    </ScrollArea>

    <Popup
        bind:open={addFriendOpen}
        onOpenChange={(open) => {
            if (open) {
                addFriendUsername = "";
                addFriendError = null;
            }
        }}
    >
        {#snippet trigger()}
            <Dialog.Trigger
                class={[
                    buttonVariants(),
                    "absolute bottom-4 right-4 !size-12 !rounded-full shadow-md"
                ]}
            >
                <AppIcon icon={Plus} class="!size-6" />
            </Dialog.Trigger>
        {/snippet}
        <Dialog.Header>
            <Dialog.Title>Add Friend</Dialog.Title>
        </Dialog.Header>
        <form
            onsubmit={(e) => {
                e.preventDefault();
                mutAddFriend.desiredValue = addFriendUsername;
            }}
        >
            <div class="grid gap-4 py-4">
                <div>
                    <Label label="Username">
                        <Input
                            disabled={mutAddFriend.applying}
                            placeholder={data.user.username}
                            bind:value={addFriendUsername}
                        />
                    </Label>
                    {#if addFriendFullErr != null}
                        <p class="text-sm text-destructive">{addFriendFullErr}</p>
                    {/if}
                </div>
            </div>
            <Dialog.Footer>
                <Button
                    class={["shadow-md", mutAddFriend.applying && "animate-pulse"]}
                    disabled={mutAddFriend.applying}
                    type="submit"
                >
                    Add
                </Button>
            </Dialog.Footer>
        </form>
    </Popup>
</div>

<PopupDetailsFriend
    bind:open={detailsDialog.open}
    user={detailsDialog.user}
    onremovefriend={async () => {
        await invalidateAll();
        detailsDialog.open = false;
    }}
/>
