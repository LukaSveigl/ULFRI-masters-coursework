<script lang="ts">
    import AppIcon from "$lib/components/AppIcon.svelte";
    import Popup from "$lib/components/Popup.svelte";
    import { onRequestFriends } from "$lib/components/PopupAddToLoan.telefunc";
    import UserSelect from "$lib/components/UserSelect.svelte";
    import { Button } from "$lib/shadcn/components/ui/button";
    import * as Dialog from "$lib/shadcn/components/ui/dialog";
    import { ScrollArea } from "$lib/shadcn/components/ui/scroll-area";
    import { sortUsers } from "$lib/sort";
    import { Plus } from "lucide-svelte";
    import { SvelteSet } from "svelte/reactivity";

    let {
        open = $bindable(false),
        disabled = false,
        extraUsers,
        presentIds,
        onconfirm
    }: {
        open?: boolean;
        disabled?: boolean;
        extraUsers: Readonly<
            {
                id: string;
                displayName: string;
                avatar: boolean;
                avatarModified: Date;
                username: string;
            }[]
        >;
        presentIds: Set<string>;
        onconfirm: (ids: string[]) => void;
    } = $props();

    function uniqueUsers<T extends { readonly id: string }>(users: T[]) {
        const encountered = new Set<string>();
        const ret: T[] = [];
        for (const user of users) {
            if (encountered.has(user.id)) continue;
            encountered.add(user.id);
            ret.push(user);
        }
        return ret;
    }

    let selected = $state(new SvelteSet<string>());
</script>

<Popup
    bind:open
    onOpenChange={(open) => {
        if (open) {
            selected.clear();
        }
    }}
>
    {#snippet trigger()}
        <Dialog.Trigger>
            {#snippet child({ props })}
                <Button {...props} variant="outline" {disabled}>
                    <AppIcon icon={Plus} class="!size-6" />
                    Add
                </Button>
            {/snippet}
        </Dialog.Trigger>
    {/snippet}
    <Dialog.Header>
        <Dialog.Title>Add Extra Users</Dialog.Title>
    </Dialog.Header>
    <ScrollArea orientation="vertical" class="h-96">
        {#await onRequestFriends() then gotUsers}
            {@const users = uniqueUsers([...gotUsers, ...extraUsers])
                .filter((x) => !presentIds.has(x.id))
                .toSorted(sortUsers)}
            <UserSelect bind:selected {users} {disabled} />
        {/await}
    </ScrollArea>
    <hr />
    <Button
        class="shadow-md"
        onclick={() => {
            onconfirm(selected.values().toArray());
            open = false;
        }}
    >
        Add
    </Button>
</Popup>
